import asyncio
from typing import List, Optional

from core.document_splitting.base_document_splitter import BaseDocumentSplitter
from core.llms.base_llm import BaseLLM
from core.summarization.base_chunked_summarizer import BaseChunkedSummarizer
from utils.prompts import load_jinja_prompt_template


class IncrementalUpdatingSummarizer(BaseChunkedSummarizer):
    def __init__(
        self,
        splitter: BaseDocumentSplitter,
        llm: Optional[BaseLLM] = None,
        batch_size: int = 2,
        summarize_all_chunks_first: bool = True,
    ):
        super().__init__(splitter, batch_size, llm)

        self.summarize_all_chunks_first = summarize_all_chunks_first

        self.initial_summary_system_template = load_jinja_prompt_template(
            "prompts/summarization/incremental_update/initial_summary.system.j2"
        )
        self.initial_summary_template = load_jinja_prompt_template(
            "prompts/summarization/incremental_update/initial_summary.j2"
        )

        self.update_summary_system_template = load_jinja_prompt_template(
            "prompts/summarization/incremental_update/update_summary.system.j2"
        )
        self.update_summary_template = load_jinja_prompt_template(
            "prompts/summarization/incremental_update/update_summary.j2"
        )

        self.compressing_prompt_system_template = load_jinja_prompt_template(
            "prompts/summarization/incremental_update/compress_summary.system.j2"
        )
        self.compress_summary_template = load_jinja_prompt_template(
            "prompts/summarization/incremental_update/compress_summary.j2"
        )

    async def _asummarize_chunks(
        self, chunks: List[str], focus_topic: Optional[str] = None
    ) -> str:
        # Initial summary from the first chunk
        if self.summarize_all_chunks_first:
            chunks = await asyncio.gather(
                *[self._asummarize_chunk(c, focus_topic) for c in chunks]
            )
            current_summary = chunks[0]
        else:
            current_summary = await self._asummarize_chunk(chunks[0])

        # Incrementally update the summary with the remaining chunks
        for i in range(1, len(chunks)):
            current_summary = await self.llm.agenerate(
                prompt=self.update_summary_template.render(
                    summary=current_summary,
                    new_chunk=chunks[i],
                    chunk_number=i,
                    total_chunk_number=len(chunks),
                ),
                system=self.update_summary_system_template.render(
                    focus_topic=focus_topic,
                ),
            )

        return current_summary

    async def _asummarize_chunk(
        self, text: str, focus_topic: Optional[str] = None
    ) -> str:
        async with self.semaphore:
            return await self.llm.agenerate(
                prompt=self.initial_summary_template.render(chunk_text=text),
                system=self.initial_summary_system_template.render(
                    focus_topic=focus_topic
                ),
            )

