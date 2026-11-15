import abc
from abc import abstractmethod
from typing import Optional, List

from pydantic.json_schema import JsonSchemaValue

from models.llms import Message


class BaseLLM(abc.ABC):
    def __init__(self, model: str):
        self.model = model

    async def agenerate(
        self,
        prompt: str,
        system: Optional[str] = None,
        format: Optional[JsonSchemaValue] = None,
    ) -> str:
        messages = []
        if system is not None:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return await self.achat(messages, format)

    @abstractmethod
    async def achat(
        self,
        messages: List[Message],
        format: Optional[JsonSchemaValue] = None,
    ) -> str:
        pass
