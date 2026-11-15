import time
import asyncio
from collections import deque
from typing import Optional, List, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

import tiktoken
from pydantic.json_schema import JsonSchemaValue

from core.llms.base_llm import BaseLLM
from models.llms import Message
from openai import AsyncAzureOpenAI, omit  # noqa


# PTU Reference:
# https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/provisioned-throughput-onboarding


class AzureLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        max_requests_per_minute: int = 60 * 5,  # max 5 requests per second
        max_tokens_per_minute: int = 14_900 * 14,
    ):
        super().__init__(model)
        self.client = AsyncAzureOpenAI(azure_deployment=model, max_retries=20)
        self.timezone = ZoneInfo("Europe/Berlin")

        # Rate Limiting Configuration
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute

        # State tracking
        self.request_logs = deque()
        self.token_logs = deque()
        self.current_token_sum = 0

        self.__tokenizer = None


    @property
    def tokenizer(self):
        if not self.__tokenizer:
            self.__tokenizer = tiktoken.encoding_for_model(self.model)
        return self.__tokenizer

    def _ensure_outside_business_hours(self):
        now = datetime.now(self.timezone)
        is_workday = now.weekday() < 5

        start_business_hour = 5
        end_business_hour = 18
        is_business_hours = start_business_hour <= now.hour < end_business_hour

        if is_workday and is_business_hours:
            raise PermissionError(
                f"Azure usage is restricted during business hours "
                f"({start_business_hour}:00-{end_business_hour}:00 Workdays). "
                f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )

    def _count_user_tokens(self, messages: List[Message]) -> int:
        user_text = "".join(
            [m.content for m in messages if m.role == "user" and m.content]
        )
        return len(self.tokenizer.encode(user_text))

    async def _wait_for_rate_limits(self, new_token_count: int):
        # Edge case: If a request is larger than the limit, it can never be send
        if new_token_count > self.max_tpm:
            raise ValueError(
                f"Single user request ({new_token_count} tokens) exceeds the "
                f"maximum per-minute limit ({self.max_tpm})."
            )

        while True:
            now = time.time()
            window_start = now - 60

            # Prune old logs
            while self.request_logs and self.request_logs[0] < window_start:
                self.request_logs.popleft()
            while self.token_logs and self.token_logs[0][0] < window_start:
                ts, count = self.token_logs.popleft()
                self.current_token_sum -= count

            # Check Request Limit (RPM)
            if len(self.request_logs) >= self.max_rpm:
                oldest_request_time = self.request_logs[0]
                wait_time = (oldest_request_time + 60) - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time + 0.1)
                continue

            # Check Token Limit (TPM)
            if self.current_token_sum + new_token_count > self.max_tpm:
                oldest_token_time = self.token_logs[0][0]
                wait_time = (oldest_token_time + 60) - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time + 0.1)
                continue

            # Record usage
            self.request_logs.append(now)
            self.token_logs.append((now, new_token_count))
            self.current_token_sum += new_token_count
            break

    async def achat(
        self,
        messages: List[Message],
        format: Optional[JsonSchemaValue] = None,
    ) -> str:
        self._ensure_outside_business_hours()

        # Count tokens (User messages only)
        token_usage = self._count_user_tokens(messages)

        # Wait if necessary
        await self._wait_for_rate_limits(token_usage)

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

        response = await self.client.chat.completions.create(
            model=self.model,
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
        self._ensure_outside_business_hours()

        messages = []
        if system is not None:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        # Count tokens (User messages only)
        token_usage = self._count_user_tokens(messages)
        await self._wait_for_rate_limits(token_usage)

        openai_messages = [message.model_dump() for message in messages]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            logprobs=True,
        )
        return [
            log.model_dump()
            for log in response.choices[0].logprobs.content[0].top_logprobs # noqa
        ]
