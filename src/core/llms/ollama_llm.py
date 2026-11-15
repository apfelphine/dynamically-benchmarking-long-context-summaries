import asyncio
import os
from typing import List, Optional

import transformers
from httpx import RemoteProtocolError
from ollama import AsyncClient, ChatResponse
from pydantic.json_schema import JsonSchemaValue

from core.llms.base_llm import BaseLLM
from models.llms import Message


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        tokenizer_name: str,
        num_ctx: int,
        num_gpu: int,
        temperature: Optional[float] = None,
        host: Optional[str] = None,
    ):
        super().__init__(model)
        if not host:
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

        self.temperature = temperature
        self.client = AsyncClient(host=host, verify=False, timeout=60 * 35)
        self.num_gpu = num_gpu
        self.num_ctx = num_ctx
        self.__tokenizer_name = tokenizer_name
        self.__tokenizer = None

    @property
    def tokenizer(self):
        if not self.__tokenizer:
            self.__tokenizer = transformers.AutoTokenizer.from_pretrained(self.__tokenizer_name)
        return self.__tokenizer

    async def achat(
        self, messages: List[Message], format: Optional[JsonSchemaValue] = None
    ):
        messages_dict = [m.model_dump() for m in messages]
        tokens = self.tokenizer.apply_chat_template(
            messages_dict,
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(tokens) > self.num_ctx:
            raise Exception(
                f"Input exceeds context window ({len(tokens)} total, {self.num_ctx} max)."
            )
        remaining_attempts = 10
        while True:
            try:
                response: ChatResponse = await self.client.chat(
                    model=self.model,
                    options={
                        "num_ctx": self.num_ctx,
                        "num_gpu": self.num_gpu,
                        "num_predict": self.num_ctx - len(tokens),
                        "temperature": self.temperature,
                    },
                    messages=messages_dict,
                    format=format,
                )
                break
            except RemoteProtocolError as e:
                remaining_attempts -= 1
                if remaining_attempts == 0:
                    raise e
                await asyncio.sleep((10 - remaining_attempts) * 10)
        if response.done_reason == "length":
            raise Exception("Generation stopped. Context window reached.")
        return response.message.content
