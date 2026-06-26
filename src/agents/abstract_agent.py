from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage

from agents.llm_clients.abstract_llm_client import AbstractLLMClient


class AbstractAgent(ABC):

    def __init__(self, llm_client: AbstractLLMClient) -> None:
        self._llm_client: AbstractLLMClient = llm_client

    @abstractmethod
    async def invoke(self, messages: list[BaseMessage]) -> BaseMessage: ...
