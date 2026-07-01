import os
import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from openai import RateLimitError

from agents.llm_clients.abstract_llm_client import AbstractLLMClient


class L3SLLMClient(AbstractLLMClient):
    """LLM client for the L3S / LLMHub vLLM endpoint."""

    def __init__(
        self,
        model: str = "ollama/qwen3.6:27b",
        base_url: str = "https://inference.kbs.uni-hannover.de/v1",  # "https://brrr.kbs.uni-hannover.de/v1",
        temperature: float = 0.7,
    ) -> None:
        self._model = model
        self._base_url = base_url
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
        api_key = os.getenv("LLMHUB_APIKEY")
        if not api_key:
            raise ValueError("LLMHUB_APIKEY environment variable not set")
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url=self._base_url,
            temperature=self._temperature,
        )
        return self._llm

    async def invoke_async(self, messages: list[Any]) -> Any:
        llm = self._create_llm()
        # loop = asyncio.get_running_loop()
        try:
            return await llm.ainvoke(messages)
        except RateLimitError as e:
            self._rate_limited = True
            self._retry_log.append({"exhausted": True, "error": str(e)[:300]})
            raise
