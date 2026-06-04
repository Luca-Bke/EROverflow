"""Terminal Bench Shell v1 purple agent.

Implements the terminal-bench-shell-v1 protocol: receives a task instruction,
issues shell commands one at a time via exec_request, and signals completion
with final. State persists across turns via the shared Agent instance per context_id.
"""

import os
import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable, tracing_context

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from openai import OpenAI

SYSTEM_PROMPT = """\
You are a terminal agent solving command-line tasks in a live shell environment.

You will receive messages as JSON. Respond ONLY with valid JSON — either:
  {"kind": "exec_request", "command": "<shell command>", "timeout": 30}
or when the task is complete:
  {"kind": "final"}

Rules:
- Never use interactive commands (vim, nano, less, ssh -t, top, htop, etc.)
- Always use non-interactive flags: apt-get -y, git --no-pager, python -c, etc.
- Verify your work before sending final (run the test or check the output)
- If a command fails, diagnose and try a different approach
- Maximum 30 commands total
- Do not include any text outside the JSON object
"""

class TerminalBenchAgent:
    """Purple agent for Terminal Bench 2.0.

    Maintains per-session conversation history across A2A turns.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = os.getenv("ACADEMICCLOUD_MODEL",
                                "meta-llama-3.1-8b-instruct")
        self._base_url = os.getenv("ACADEMICCLOUD_ENDPOINT",
                                   "https://chat-ai.academiccloud.de/v1")
        self._trace_enabled = bool(os.getenv("LANGSMITH_API_KEY"))
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "EROverflow-terminal-bench"

        self._llm: ChatOpenAI | None = None
        self._llm_client: OpenAI | None = None
        self._history: list[Any] = []
        self._turn_count = 0
        self._temperature = 0.7
        self._boundary_logged = False

    @traceable(name="green_purple_boundary", run_type="chain")
    def _trace_boundary(self, input_text: str, response_text: str) -> dict[str, str]:
        if self._boundary_logged:
            return {"skipped": "true"}
        self._boundary_logged = True
        return {
            "event": "trace_boundary",
            "source": "green_agent",
            "via": "terminal_bench",
            "target": "green_agent",
            "start_payload": input_text,
            "end_payload": response_text,
        }

    def _create_llm(self) -> ChatOpenAI:
        """Create or get the ChatOpenAI instance for AcademicCloud."""
        if self._llm:
            return self._llm

        api_key = os.getenv("ACADEMICCLOUD_API_KEY")

        if not api_key:
            raise ValueError(
                "ACADEMICCLOUD_API_KEY environment variable not set")

        self._llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url=self._base_url,
            temperature=self._temperature,
            tags=["eroverflow", "terminal-bench"],
            metadata={"agent": "terminal_bench", "provider": "academiccloud"},
        )
        return self._llm
    @traceable(run_type="llm")
    async def _invoke_llm_async(self, messages: list[Any]) -> str:
        """Invoke the LLM in a non-blocking way using asyncio."""
        llm = self._create_llm()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: llm.invoke(messages))
        return result
    @traceable
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        self._history.append(
            HumanMessage(content=input_text)
        )

        print(f"Received task: {input_text}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Turn {self._turn_count}: thinking...")
        )

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + self._history
        result = await self._invoke_llm_async(messages)
        response_text = getattr(result, "content", str(result))
        self._history.append(AIMessage(content=response_text))

        print(f"LLM response: {response_text}")

        # add response as an artifact
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="Response",
        )

        # if the response indicates the task is complete, send final status
        response_msg = updater.new_agent_message(
            parts=[Part(root=TextPart(text=response_text))]
        )

        with tracing_context(enabled=self._trace_enabled):
            self._trace_boundary(input_text, response_text)

        await updater.submit(response_msg)

        await updater.update_status(
            TaskState.completed, new_agent_text_message(
                "Completed requested task.")
        )


__all__ = ["TerminalBenchAgent"]
