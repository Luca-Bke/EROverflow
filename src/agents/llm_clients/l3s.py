import os
from typing import Any

from langchain_openai import ChatOpenAI
from openai import RateLimitError

from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.llm_clients.retry import ainvoke_with_backoff


class L3SLLMClient(AbstractLLMClient):
    """LLM client for the L3S / LLMHub vLLM endpoint."""

    def __init__(
        self,
        model: str = "ollama/qwen3.6:27b",
        base_url: str = "https://inference.kbs.uni-hannover.de/v1",  # "https://brrr.kbs.uni-hannover.de/v1",
        temperature: float = 0.7,
        timeout: float = 120.0,
        backoff_enabled: bool = True,
        backoff_max_retries: int = 4,
        backoff_base_delay: float = 5.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._timeout = timeout
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
        api_key = os.getenv("LLMHUB_APIKEY")
        if not api_key:
            raise ValueError("LLMHUB_APIKEY environment variable not set")
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url=self._base_url,
            temperature=self._temperature,
            timeout=self._timeout,
            max_retries=0,  # our own backoff is the single retry source
        )
        return self._llm

    async def invoke_async(self, messages: list[Any]) -> Any:
        llm = self._create_llm()
        max_retries = self._backoff_max_retries if self._backoff_enabled else 1
        try:
            return await ainvoke_with_backoff(
                llm,
                messages,
                max_retries=max_retries,
                base_delay=self._backoff_base_delay,
                retry_log=self._retry_log,
            )
        except RateLimitError:
            # Keep the tracing flag meaningful when we ultimately give up on 429s.
            self._rate_limited = True
            raise
