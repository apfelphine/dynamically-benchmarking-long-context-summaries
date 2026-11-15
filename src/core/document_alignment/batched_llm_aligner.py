import asyncio
from typing import List, Dict

import numpy as np
import tqdm

from pydantic import BaseModel, Field

from core.document_alignment.base_document_aligner import BaseDocumentAligner
from core.llms.base_llm import BaseLLM
from models.document_splitting import Chunk
from utils.prompts import load_jinja_prompt_template


class SummaryChunkResponseModel(BaseModel):
    chunk_idx: int
    supported_information: str = Field(...)
    unsupported_information: str = Field(...)
    score: int = Field(..., ge=0, le=100)

    class Config:
        extra = "forbid"


class ResponseModel(BaseModel):
    results: List[SummaryChunkResponseModel]

    class Config:
        extra = "forbid"


class BatchedLLMAligner(BaseDocumentAligner):
    """Groups all summary chunks per document chunk in one request"""

    def __init__(
        self,
        llm: BaseLLM,
        batch_size: int = 8,
        all_summary_chunks: bool = True,
    ):
        self.event_loop = asyncio.get_event_loop()
        self.batch_size = batch_size
        self.llm = llm
        self.all_summary_chunks = all_summary_chunks

        self.system_prompt = load_jinja_prompt_template(
            "prompts/document_alignment/batched_llm/align.system.j2"
        )
        self.user_prompt = load_jinja_prompt_template(
            "prompts/document_alignment/row_based_llm/align.j2"
        )

    async def acompare(self, document_chunk: Chunk, summary_chunk: Chunk) -> float:
        raise NotImplementedError()

    async def _request(
        self,
        document_index: int,
        document_chunk: Chunk,
        summary_indices: List[int],
        summary_chunks: List[Chunk],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, float]:
        prompt = self.user_prompt.render(
            document_chunk=document_chunk, summary_chunks=summary_chunks
        )

        retries = 3
        for idx in range(retries):

            async with semaphore:
                response = await self.llm.agenerate(
                    system=self.system_prompt.render(),
                    prompt=prompt,
                    format=ResponseModel.model_json_schema(),
                )
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                validated_response = ResponseModel.model_validate_json(response)
                break
            except ValueError:
                if idx == retries - 1:
                    raise

        scores = {}

        for result in validated_response.results:
            summary_chunk_idx = result.chunk_idx
            if summary_chunk_idx >= len(summary_indices):
                continue
            key = f"{document_index}.{summary_indices[summary_chunk_idx]}"
            score = result.score / 100
            if key in scores:
                scores[key] = (score + scores[key]) / 2
            else:
                scores[key] = score

        return scores

    async def _continue_similarity_matrix(
        self,
        document_chunks: List[Chunk],
        summary_chunks: List[Chunk],
        matrix: np.ndarray,
    ) -> np.ndarray:
        tasks = []
        semaphore = asyncio.Semaphore(self.batch_size)
        for i, document_chunk in enumerate(document_chunks):
            doc_row = matrix[i, :]
            if not np.any(np.isnan(doc_row)):
                continue
            sum_indices = (
                list(np.argwhere(np.isnan(doc_row)).flatten())
                if not self.all_summary_chunks
                else range(0, len(doc_row))
            )
            sum_chunks = [summary_chunks[idx] for idx in sum_indices]
            if sum_chunks:
                tasks.append(
                    asyncio.create_task(
                        self._request(
                            i, document_chunk, sum_indices, sum_chunks, semaphore
                        )
                    )
                )

        pbar = tqdm.tqdm(total=len(tasks), desc="Building similarity matrix...")
        for task in tasks:
            task.add_done_callback(lambda t: pbar.update(1))

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )

            # If there are pending tasks, it means an exception triggered the return
            if pending:
                # Manually cancel all remaining tasks
                for task in pending:
                    task.cancel()
                await asyncio.wait(pending)

                # Find the task(s) that failed and re-raise its exception
                e = None
                for task in done:
                    if task.exception():
                        e = task.exception()
                raise e

            # If we reach here, all tasks succeeded
            for task in done:
                r = task.result()
                for indices, score in r.items():
                    split_indices = indices.split(".")
                    matrix[int(split_indices[0]), int(split_indices[1])] = score

        finally:
            pbar.close()

        if np.any(np.isnan(matrix)):
            matrix[np.isnan(matrix)] = 0
        return matrix
