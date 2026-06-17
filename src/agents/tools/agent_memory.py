from collections import deque
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


class AgentMemory:
    """Conversation memory split into long-term and short-term tiers.

    Long-term (always in every LLM call):
      - system prompt
      - initial task HumanMessage
      - persistent extras (e.g. critic feedback, added via add_long_term)

    Short-term (rolling window, most recent N messages):
      - exec_result HumanMessages and AI responses

    Usage:
        memory = MemoryData(SystemMessage(content=SYSTEM_PROMPT), short_term_window=10)
        memory.set_task(HumanMessage(content=task_json))
        memory.add(HumanMessage(content=exec_result_json))
        memory.add(AIMessage(content=response_text))
        messages = memory.build_messages()  # pass to LLM
    """

    def __init__(self, system_prompt: SystemMessage, short_term_window: int = 10) -> None:
        self._system_prompt = system_prompt
        self._task_message: HumanMessage | None = None
        # Actor-authored plan, persisted in long-term so it survives the
        # rolling short-term window. Updated via set_plan (overwrites).
        self._plan: HumanMessage | None = None
        # Persistent extras sit between the task and short-term (e.g. critic feedback)
        self._long_term_extras: list[Any] = []
        # Rolls off oldest messages once maxlen is reached (window = N messages = N/2 pairs)
        self._short_term: deque[Any] = deque(maxlen=short_term_window)

    def set_task(self, message: HumanMessage) -> None:
        """Store the initial task message as long-term memory."""
        self._task_message = message

    def set_plan(self, plan: Any) -> None:
        """Store/overwrite the Actor's current plan as long-term memory.

        `plan` may be the parsed plan dict (with a "steps" or "plan" field) or a
        plain string. It is rendered into a HumanMessage that is injected into
        every LLM call so the Actor can work through it step by step.
        """
        if plan is None:
            self._plan = None
            return
        if isinstance(plan, dict):
            steps = plan.get("steps") or plan.get("plan")
        else:
            steps = plan
        if isinstance(steps, (list, tuple)):
            body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        else:
            body = str(steps)
        self._plan = HumanMessage(
            content="## Current plan (work through it step by step):\n" + body
        )

    def get_plan(self) -> HumanMessage | None:
        """Return the stored plan message, if any."""
        return self._plan

    def add(self, message: Any) -> None:
        """Append a message to the short-term rolling window."""
        self._short_term.append(message)

    def add_long_term(self, message: Any) -> None:
        """Persist a message in long-term memory (e.g. critic feedback, constraints).

        These are always included in every LLM call, between the task and short-term window.
        """
        self._long_term_extras.append(message)

    def build_messages(self) -> list[Any]:
        """Assemble the full message list for the next LLM call.

        Order: SystemMessage → task → plan → long_term_extras → short_term window
        """
        messages: list[Any] = [self._system_prompt]
        if self._task_message:
            messages.append(self._task_message)
        if self._plan:
            messages.append(self._plan)
        messages.extend(self._long_term_extras)
        messages.extend(self._short_term)
        return messages
