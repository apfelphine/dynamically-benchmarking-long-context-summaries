import os
from typing import Optional, List

import httpx
import numpy as np
from openai import AsyncOpenAI

from core.embedding.base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not api_key:
            api_key = ""
        super().__init__(model)

        headers = {}
        if os.environ.get("REMOTE_AUTH", None):
            headers = {"Authorization": os.environ.get("REMOTE_AUTH")}

        http_client = httpx.AsyncClient(headers=headers, verify=False)
        self.client = AsyncOpenAI(
            api_key=api_key, http_client=http_client, base_url=base_url
        )

    async def embed(self, input: str | List[str]):
        res = await self.client.embeddings.create(
            model=self.model,
            input=input,
        )
        return [np.array(e.embedding) for e in res.data]
