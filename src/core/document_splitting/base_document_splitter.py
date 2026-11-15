from abc import abstractmethod, ABC
from typing import List

from models.document_splitting import Chunk


class BaseDocumentSplitter(ABC):
    async def split(self, text) -> List[Chunk]:
        result = []
        for t in await self._split(text):
            stripped_t = t.strip()
            if not stripped_t:
                continue

            start_idx, end_idx = self.get_index(text, stripped_t)
            chunk = Chunk(
                text=stripped_t, text_original=t, idx_start=start_idx, idx_end=end_idx
            )
            result.append(chunk)

        return result

    @abstractmethod
    async def _split(self, text) -> List[str]:
        pass

    @staticmethod
    def get_index(text: str, chunk: str) -> tuple[int, int]:
        return text.index(chunk), text.index(chunk) + len(chunk)
