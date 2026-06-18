import asyncio
import os
from typing import Any

from langchain_openai import ChatOpenAI
from openai import RateLimitError

from agents.llms.llm_client import TerminalBenchLLMClientInterface


class AcademicCloudLLMClient(TerminalBenchLLMClientInterface):
    """Thin wrapper around ChatOpenAI for the AcademicCloud endpoint.

    Owns connection config, lazy LLM instantiation, and exponential-backoff
    retry logic for 429 responses. All other agent logic lives in the caller.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        backoff_enabled: bool | None = None,
        backoff_max_retries: int | None = None,
        backoff_base_delay: float | None = None,
    ) -> None:
        self._model = model or os.getenv(
            "ACADEMICCLOUD_MODEL", "qwen3.6-35b-a3b")
        self._base_url = base_url or os.getenv(
            "ACADEMICCLOUD_ENDPOINT", "https://chat-ai.academiccloud.de/v1"
        )
        self._temperature = temperature

        self._backoff_enabled = (
            backoff_enabled
            if backoff_enabled is not None
            else os.getenv("ENABLE_RATE_LIMIT_BACKOFF", "true").lower()
            in ("1", "true", "yes", "True")
        )
        self._backoff_max_retries = (
            backoff_max_retries
            if backoff_max_retries is not None
            else int(os.getenv("BACKOFF_MAX_RETRIES", "4"))
        )
        self._backoff_base_delay = (
            backoff_base_delay
            if backoff_base_delay is not None
            else float(os.getenv("BACKOFF_BASE_DELAY", "5.0"))
        )

        self._llm: ChatOpenAI | None = None
        self._rate_limited = False
        self._retry_log: list[dict] = []

    def rate_limited(self) -> bool:
        return self._rate_limited

    def retry_log(self) -> list[dict]:
        return self._retry_log

    def model(self) -> str:
        return self._model

    def _get_api_key(self) -> str:
        api_key = os.getenv("ACADEMICCLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "ACADEMICCLOUD_API_KEY environment variable not set")
        return api_key

    def _create_llm(self) -> ChatOpenAI:
        if self._llm:
            return self._llm
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=self._get_api_key(),
            base_url=self._base_url,
            temperature=self._temperature,
            tags=["eroverflow", "terminal-bench"],
            metadata={"agent": "terminal_bench", "provider": "academiccloud"},
        )
        return self._llm

    async def invoke_async(self, messages: list[Any]) -> Any:
        """Invoke the LLM. On 429, retries with exponential backoff (if enabled),
        then sets rate_limited and re-raises when retries are exhausted."""
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
