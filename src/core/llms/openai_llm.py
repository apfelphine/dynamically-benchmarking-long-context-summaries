from typing import Optional, List, Dict

import transformers
from pydantic.json_schema import JsonSchemaValue

from core.llms.base_llm import BaseLLM
from models.llms import Message
import httpx
from openai import AsyncOpenAI, omit
import os

open_router_base_url = "https://openrouter.ai/api/v1"
openai_base_url = "https://api.openai.com/v1"


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str,
        tokenizer_name: str,
        ctx_window: int,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        timeout: Optional[float | int] = None,
    ):
        if not api_key:
            api_key = ""
        super().__init__(model)
        self.temperature = temperature

        headers = {}
        if os.environ.get("REMOTE_AUTH", None):
            headers = {"Authorization": os.environ.get("REMOTE_AUTH")}

        http_client = httpx.AsyncClient(headers=headers, verify=False)
        self.client = AsyncOpenAI(api_key=api_key, http_client=http_client, base_url=base_url, timeout=timeout)
        self.frequency_penalty = frequency_penalty
        self.ctx_window = ctx_window

        self.__tokenizer_name = tokenizer_name
        self.__tokenizer = None

    @property
    def tokenizer(self):
        if not self.__tokenizer:
            self.__tokenizer = transformers.AutoTokenizer.from_pretrained(self.__tokenizer_name)
        return self.__tokenizer


    async def achat(
        self,
        messages: List[Message],
        format: Optional[JsonSchemaValue] = None,
    ) -> str:
        response_format = omit
        if format is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": format.get("title", "ResponseModel")
                    .replace("[", "")
                    .replace("]", ""),
                    "strict": True,
                    "schema": format,
                },
            }

        openai_messages = [message.model_dump() for message in messages]

        tokens = self.tokenizer.apply_chat_template(
            openai_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(tokens) > self.ctx_window:
            raise Exception(
                f"Input exceeds context window ({len(tokens)} total, {self.ctx_window} max)."
            )
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature if self.temperature else omit,
            frequency_penalty=self.frequency_penalty if self.frequency_penalty else omit,
            messages=openai_messages,
            response_format=response_format,
        )
        if hasattr(response, "error") and response.error:
            raise Exception(response.error["message"])

        return response.choices[0].message.content.strip()

    async def get_log_probs(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1,
        top_logprobs: int = 5,
    ) -> List[Dict]:
        messages = []
        if system is not None:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        openai_messages = [message.model_dump() for message in messages]
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=openai_messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            logprobs=True,
        )
        return [
            log.model_dump()
            for log in response.choices[0].logprobs.content[0].top_logprobs # noqa
        ]
