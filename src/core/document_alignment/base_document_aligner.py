from abc import abstractmethod, ABC
from typing import List

import numpy as np
from tqdm import tqdm

from models.document_splitting import Chunk


class BaseDocumentAligner(ABC):
    @abstractmethod
    async def acompare(self, document_chunk: Chunk, summary_chunk: Chunk) -> float:
        pass

    async def asimilarity_matrix(
        self, document_chunks: List[Chunk], summary_chunks: List[Chunk]
    ) -> np.ndarray:
        matrix = np.zeros((len(document_chunks), len(summary_chunks)))
        matrix[:] = np.nan
        return await self._continue_similarity_matrix(
            document_chunks, summary_chunks, matrix
        )

    async def _continue_similarity_matrix(
        self,
        document_chunks: List[Chunk],
        summary_chunks: List[Chunk],
        matrix: np.ndarray,
    ):
        total = len(document_chunks) * len(summary_chunks)
        desc = "Building similarity matrix..."
        with tqdm(total=total, desc=desc) as pbar:
            for i, document_chunk in enumerate(document_chunks):
                doc_desc = desc + " - Document Chunk: " + str(i)
                pbar.set_description(doc_desc)
                for j, summary_chunk in enumerate(summary_chunks):
                    pbar.set_description(doc_desc + " - Summary Chunk: " + str(j))
                    if np.isnan(matrix[i, j]):
                        matrix[i, j] = await self.acompare(
                            document_chunk, summary_chunk
                        )
                    pbar.update(1)
        return matrix
