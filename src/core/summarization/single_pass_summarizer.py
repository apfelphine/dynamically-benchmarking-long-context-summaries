from typing import Optional

from core.llms.base_llm import BaseLLM
from core.summarization.base_summarizer import BaseSummarizer
from utils.prompts import load_jinja_prompt_template


class SinglePassSummarizer(BaseSummarizer):
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
    ):
        super().__init__(llm)

        self.system_prompt_template = load_jinja_prompt_template(
            "prompts/summarization/single_pass/summarize.system.j2"
        )
        self.user_prompt_template = load_jinja_prompt_template(
            "prompts/summarization/single_pass/summarize.j2"
        )

    async def _asummarize(self, text: str, focus_topic: Optional[str] = None) -> str:
        return await self.llm.agenerate(
            prompt=self.user_prompt_template.render(text=text),
            system=self.system_prompt_template.render(focus_topic=focus_topic),
        )
