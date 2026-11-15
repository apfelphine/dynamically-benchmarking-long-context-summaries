from typing import List, Dict

import numpy as np

from core.document_alignment.base_document_aligner import BaseDocumentAligner
from utils.alignment import make_binary
from models.document_splitting import Chunk


class HybridDocumentAligner(BaseDocumentAligner):
    def __init__(
        self,
        comparer1: BaseDocumentAligner,
        comparer2: BaseDocumentAligner,
        config: Dict,
    ):
        self.comparer1 = comparer1
        self.comparer2 = comparer2
        self.config = config

    async def acompare(self, document_chunk: Chunk, summary_chunk: Chunk) -> float:
        raise NotImplementedError()

    async def execute_first_comparer(
        self, document_chunks: List[Chunk], summary_chunks: List[Chunk]
    ):
        return await self.comparer1.asimilarity_matrix(document_chunks, summary_chunks)

    async def execute_second_comparer(
        self,
        document_chunks: List[Chunk],
        summary_chunks: List[Chunk],
        sim_matrix: np.ndarray,
    ):
        binary_sim_matrix = make_binary(sim_matrix, **self.config).astype(float)
        binary_sim_matrix[binary_sim_matrix == 1.0] = np.nan

        return await self.comparer2._continue_similarity_matrix(
            document_chunks, summary_chunks, binary_sim_matrix
        )

    async def asimilarity_matrix(
        self, document_chunks: List[Chunk], summary_chunks: List[Chunk]
    ) -> np.ndarray:
        sim_matrix = await self.execute_first_comparer(document_chunks, summary_chunks)
        return await self.execute_second_comparer(
            document_chunks, summary_chunks, sim_matrix
        )
