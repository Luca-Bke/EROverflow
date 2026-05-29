"""Terminal Bench Shell v1 purple agent.

Implements the terminal-bench-shell-v1 protocol: receives a task instruction,
issues shell commands one at a time via exec_request, and signals completion
with final. State persists across turns via the shared Agent instance per context_id.
"""

import json
import os
import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

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
        self._model = model or os.getenv("AGENT_LLM", "DeepSeek-R1-Distill-Llama-70B")
        self._llm: ChatOpenAI | None = None
        self._history: list[Any] = []
        self._turn_count = 0

    def _create_llm(self) -> ChatOpenAI:
        if self._llm:
            return self._llm
        api_key = os.getenv("ACADEMICCLOUD_API_KEY")
        if not api_key:
            raise ValueError("ACADEMICCLOUD_API_KEY environment variable not set")
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url="https://chat-ai.academiccloud.de/v1",
            temperature=0.0,
        )
        return self._llm

    async def _invoke_async(self, messages: list[Any]) -> str:
        llm = self._create_llm()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: llm.invoke(messages))
        return getattr(result, "content", str(result)).strip()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            payload = json.loads(input_text)
        except json.JSONDecodeError:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Expected JSON, got: {input_text[:200]}")
            )
            return

        kind = payload.get("kind")

        if kind == "task":
            # First turn: new task arriving
            self._history = []
            self._turn_count = 0
            instruction = payload.get("instruction", "")
            self._history.append(
                HumanMessage(content=json.dumps({
                    "kind": "task",
                    "instruction": instruction,
                }))
            )
        elif kind == "exec_result":
            # Subsequent turns: result of the last command
            self._history.append(HumanMessage(content=json.dumps(payload)))
        else:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Unknown message kind: {kind}")
            )
            return

        self._turn_count += 1

        # Hard stop to avoid runaway loops
        if self._turn_count > 30:
            response_text = json.dumps({"kind": "final"})
        else:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Turn {self._turn_count}: thinking...")
            )
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + self._history
            raw = await self._invoke_async(messages)

            # Ensure the response is valid JSON; fall back to final if not
            try:
                parsed = json.loads(raw)
                response_text = json.dumps(parsed)
            except json.JSONDecodeError:
                # LLM returned prose — treat as final answer
                response_text = json.dumps({"kind": "final"})

        # Record assistant turn in history
        self._history.append(AIMessage(content=response_text))

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="Response",
        )
        response_msg = updater.new_agent_message(
            parts=[Part(root=TextPart(text=response_text))]
        )
        await updater.submit(response_msg)
        await updater.update_status(
            TaskState.completed, new_agent_text_message("Done")
        )


__all__ = ["TerminalBenchAgent"]
