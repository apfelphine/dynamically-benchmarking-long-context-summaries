import abc
from abc import abstractmethod
from typing import List


class BaseEmbedding(abc.ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def embed(
        self,
        input: str | List[str],
    ):
        pass
