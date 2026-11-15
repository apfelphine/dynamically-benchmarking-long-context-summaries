import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List

from core.document_splitting.base_document_splitter import BaseDocumentSplitter
from core.llms.base_llm import BaseLLM
from core.summarization.base_summarizer import BaseSummarizer


class BaseChunkedSummarizer(BaseSummarizer, ABC):
    def __init__(
        self,
        splitter: BaseDocumentSplitter,
        batch_size: int = 2,  # Max concurrent requests
        llm: Optional[BaseLLM] = None,
    ) -> None:
        super().__init__(llm)
        self.semaphore = asyncio.Semaphore(batch_size)
        self.splitter = splitter

    async def _split(self, text: str) -> List[str]:
        return [c.text for c in await self.splitter.split(text)]

    async def _asummarize(self, text: str, focus_topic: Optional[str] = None) -> str:
        return await self._asummarize_chunks(await self._split(text), focus_topic)

    @abstractmethod
    async def _asummarize_chunks(
        self, chunks: List[str], focus_topic: Optional[str] = None
    ) -> str:
        pass
