import asyncio
import os
from typing import Any

from langchain_openai import ChatOpenAI
from openai import RateLimitError

from agents.llm_clients.abstract_llm_client import AbstractLLMClient


class AcademicCloudLLMClient(AbstractLLMClient):
    """LLM client for the AcademicCloud endpoint with exponential-backoff
    retry."""

    def __init__(
        self,
        model: str = "qwen3.6-35b-a3b",
        base_url: str = "https://chat-ai.academiccloud.de/v1",
        temperature: float = 0.7,
        backoff_enabled: bool = True,
        backoff_max_retries: int = 4,
        backoff_base_delay: float = 5.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._backoff_enabled = backoff_enabled
        self._backoff_max_retries = backoff_max_retries
        self._backoff_base_delay = backoff_base_delay
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
        api_key = os.getenv("ACADEMICCLOUD_API_KEY")
        if not api_key:
            raise ValueError("ACADEMICCLOUD_API_KEY environment variable not set")
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url=self._base_url,
            temperature=self._temperature,
            tags=["eroverflow", "terminal-bench"],
            metadata={"agent": "terminal_bench", "provider": "academiccloud"},
        )
        return self._llm

    async def invoke_async(self, messages: list[Any]) -> Any:
        """Invoke the LLM, retrying with exponential backoff on 429s."""
        llm = self._create_llm()
        loop = asyncio.get_running_loop()
        max_attempts = self._backoff_max_retries if self._backoff_enabled else 1

        for attempt in range(max_attempts):
            try:
                return await loop.run_in_executor(None, lambda: llm.invoke(messages))
            except RateLimitError as e:
                if self._backoff_enabled and attempt < max_attempts - 1:
                    delay = self._backoff_base_delay * (2 ** attempt)
                    self._retry_log.append({
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "delay_seconds": delay,
                        "error": str(e)[:300],
                    })
                    print(
                        f"Rate limit hit (attempt {attempt + 1}/{max_attempts}),"
                        f" retrying in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self._rate_limited = True
                    self._retry_log.append({
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "exhausted": True,
                        "error": str(e)[:300],
                    })
                    raise
