"""Actor agent that uses native LLM tool calls to propose shell commands."""

from typing import override

from langchain_core.messages import BaseMessage
from langsmith import traceable

from agents.abstract_agent import AbstractAgent
from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.terminal_bench_supplementary.utils import TimeTracer


class ActorAgent(AbstractAgent):
    """Generates an exec_request candidate via tool call.

    The Actor receives a list of tools (currently just `execute_command`) and
    either:
      - emits a tool call to run a shell command, or
      - emits a plain text response to signal task completion (finalize).
    """

    def __init__(self, llm_client: AbstractLLMClient) -> None:
        super().__init__(llm_client)

    @override
    @traceable(name="Actor", run_type="chain")
    @TimeTracer.timed("ActorAgent.invoke")
    async def invoke(
        self,
        messages: list[BaseMessage],
        tools: list[dict],
    ) -> BaseMessage:
        """Invoke the Actor LLM with tool definitions.

        Returns an AIMessage that either carries ``tool_calls`` (for
        ``execute_command``) or plain text content (for finalize).
        """
        return await self._llm_client.invoke_with_tools_async(
            messages, tools
        )
