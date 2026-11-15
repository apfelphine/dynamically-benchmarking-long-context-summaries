from abc import abstractmethod, ABC
from typing import List, Optional

from core.llms.base_llm import BaseLLM


class BaseSummarizer(ABC):
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def asummarize_multi(
        self, texts: List[str], focus_topic: Optional[str] = None
    ) -> str:
        return await self.asummarize("\n\n".join(texts), focus_topic)

    async def asummarize(self, text: str, focus_topic: Optional[str] = None) -> str:
        if not text.strip():
            raise Exception("No text to summarize")

        return await self._asummarize(text, focus_topic)

    @abstractmethod
    async def _asummarize(self, text: str, focus_topic: Optional[str] = None) -> str:
        pass
