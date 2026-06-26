from langchain_core.messages import BaseMessage

from agents.abstract_agent import AbstractAgent
from agents.llm_clients.abstract_llm_client import AbstractLLMClient


class ActorAgent(AbstractAgent):
    """Generates an exec_request candidate from the task formulation and conversation history."""

    def __init__(self, llm_client: AbstractLLMClient) -> None:
        super().__init__(llm_client)

    async def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        return await self._llm_client.invoke_async(messages)
