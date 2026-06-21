import os
import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from openai import RateLimitError

from agents.llms.llm_client import TerminalBenchLLMClientInterface


class OpenRouterLLMClient(TerminalBenchLLMClientInterface):
    """Agent wrapper around OpenRouter-backed ChatOpenAI.

    Usage: instantiate and call `await agent.run(message, updater)` from an
    async context (matches the signature used in `src/agent.py`).
    """

    def __init__(self, model: str | None = None,
                 temperature: float = 0.7) -> None:
        self._model = model or "openai/gpt-oss-120b:free"
        self._temperature = temperature
        self._llm: ChatOpenAI | None = None
        self._rate_limited = False
        self._retry_log: list[dict] = []

    def rate_limited(self) -> bool:
        return self._rate_limited

    def retry_log(self) -> list[dict]:
        return self._retry_log

    def model(self) -> str:
        return self._model

    def _create_llm(self) -> ChatOpenAI:
        if self._llm:
            return self._llm

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self._llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=self._temperature,
        )
        return self._llm

    async def invoke_async(self, messages: list[Any]) -> Any:
        llm = self._create_llm()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: llm.invoke(messages))
        except RateLimitError as e:
            self._rate_limited = True
            self._retry_log.append({"exhausted": True, "error": str(e)[:300]})
            raise
