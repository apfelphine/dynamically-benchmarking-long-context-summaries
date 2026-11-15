import asyncio
from typing import List, Optional

from core.document_splitting.base_document_splitter import BaseDocumentSplitter
from core.llms.base_llm import BaseLLM
from core.summarization.base_chunked_summarizer import BaseChunkedSummarizer
from utils.prompts import load_jinja_prompt_template


class HierarchicalMergingSummarizer(BaseChunkedSummarizer):
    def __init__(
        self,
        splitter: BaseDocumentSplitter,
        llm: Optional[BaseLLM] = None,
        batch_size: int = 2,  # Max concurrent requests
    ):
        super().__init__(splitter, batch_size, llm)

        self.chunk_summary_system_prompt = load_jinja_prompt_template(
            "prompts/summarization/hierarchical_merge/chunk_summary.system.j2"
        )
        self.chunk_summary_template = load_jinja_prompt_template(
            "prompts/summarization/hierarchical_merge/chunk_summary.j2"
        )

        self.merge_summaries_system_prompt = load_jinja_prompt_template(
            "prompts/summarization/hierarchical_merge/merge_summaries.system.j2"
        )
        self.merge_template = load_jinja_prompt_template(
            "prompts/summarization/hierarchical_merge/merge_summaries.j2"
        )

    async def _asummarize_chunk(
        self, chunk: str, focus_topic: Optional[str] = None
    ) -> str:
        async with self.semaphore:
            return await self.llm.agenerate(
                system=self.chunk_summary_system_prompt.render(focus_topic=focus_topic),
                prompt=self.chunk_summary_template.render(chunk_text=chunk),
            )

    async def _amerge_summaries(
        self, summaries: List[str], focus_topic: Optional[str] = None
    ) -> str:
        async with self.semaphore:
            if len(summaries) != 2:
                raise ValueError("Merging requires exactly two summaries.")

            return await self.llm.agenerate(
                system=self.merge_summaries_system_prompt.render(
                    focus_topic=focus_topic
                ),
                prompt=self.merge_template.render(
                    summary1=summaries[0], summary2=summaries[1]
                ),
            )

    @staticmethod
    async def pass_through(s):
        return s

    async def _asummarize_chunks(
        self, chunks: List[str], focus_topic: Optional[str] = None
    ) -> str:
        if len(chunks) == 1:
            # No need for hierarchical merging if there's only one chunk
            return await self._asummarize_chunk(chunks[0], focus_topic)

        # Summarize all chunks in parallel
        summarization_tasks = [
            self._asummarize_chunk(chunk, focus_topic) for chunk in chunks
        ]
        current_summaries = await asyncio.gather(*summarization_tasks)

        while len(current_summaries) > 1:
            merge_tasks = []
            # Create tasks to merge summaries
            for i in range(0, len(current_summaries), 2):
                if i + 1 < len(current_summaries):
                    task = self._amerge_summaries(
                        [current_summaries[i], current_summaries[i + 1]], focus_topic
                    )
                    merge_tasks.append(task)
                else:
                    # In case of uneven chunk numbers there won't be any chunk to merge with,
                    # just pass through this chunk
                    merge_tasks.append(self.pass_through(current_summaries[i]))

            current_summaries = await asyncio.gather(*merge_tasks)

        return current_summaries[0]
