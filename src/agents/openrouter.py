"""OpenRouter integration exposing an `OpenRouterAgent` compatible with the
agent interface used in `src/agent.py`.

This module provides `OpenRouterAgent` with an async `run(message, updater)`
method so it can be plugged into the existing task/updater workflow.
"""

import os
import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message


class OpenRouterAgent:
    """Agent wrapper around OpenRouter-backed ChatOpenAI.

    Usage: instantiate and call `await agent.run(message, updater)` from an
    async context (matches the signature used in `src/agent.py`).
    """

    def __init__(self, model: str | None = None, temperature: float = 0.7) -> None:
        self._model = model or "openai/gpt-oss-120b:free"
        self._temperature = temperature
        self._llm: ChatOpenAI | None = None

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

    async def _invoke_async(self, messages: list[Any]) -> Any:
        llm = self._create_llm()
        loop = asyncio.get_running_loop()
        # ChatOpenAI.invoke is synchronous; run it in a thread to avoid blocking
        return await loop.run_in_executor(None, lambda: llm.invoke(messages))

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Handle incoming `message` and report results via `updater`.

        This method follows the same contract as `Agent.run` in `src/agent.py`.
        """
        input_text = get_message_text(message)

        await updater.update_status(TaskState.working, new_agent_text_message("Thinking..."))

        system = SystemMessage(
            content=(
                "You are a Nurse in an Emergency Room in a Hospital. You will be"
                " presented with several patients showing symptoms. You must triage"
                " them according to the Manchester Code."
            )
        )
        user = HumanMessage(content=input_text)

        try:
            result = await self._invoke_async([system, user])
            # Extract text from the model response
            agent_text = getattr(result, "content", str(result))

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=agent_text))],
                name="OpenRouter Response",
            )

            response = updater.new_agent_message(
                parts=[Part(root=TextPart(text=agent_text))]
            )

            await updater.submit(response)
            await updater.update_status(TaskState.completed,
                                        new_agent_text_message("Completed"))
        except Exception as e:
            await updater.update_status(TaskState.failed,
                                        new_agent_text_message(str(e)))


__all__ = ["OpenRouterAgent"]
