from collections import deque
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.terminal_bench_supplementary.pipeline_messages import (
    CriticFeedbackMessage,
    HumanTaskMessage,
    TaskFormulationMessage,
)
from agents.terminal_bench_supplementary.utils import apply_message_label


class AgentMemory:
    """Shared memory for a planner-actor-critic agent pipeline.

    Holds the three agent system prompts and the shared state they read/write:
      - plan: planner's current execution plan (one message, overwritten on update)
      - task_formulation: planner's sub-task instruction for the actor (one message)
      - chat_history: full conversation history (tool calls, tool responses, plan)
      - pending_tool_call: actor's current tool call awaiting critic verdict

    Call build_planner_messages(), build_actor_messages(), or
    build_critic_messages() to assemble the prompt for each respective agent.
    """

    def __init__(
        self,
        planner_system_prompt: SystemMessage,
        actor_system_prompt: SystemMessage,
        critic_system_prompt: SystemMessage,
        short_term_window: int = 10,
    ) -> None:
        self._planner_system_prompt = planner_system_prompt
        self._actor_system_prompt = actor_system_prompt
        self._critic_system_prompt = critic_system_prompt
        self._task: HumanMessage | None = None
        self._plan: HumanMessage | None = None
        self._subtask_formulation: HumanMessage | None = None
        self._pending_tool_call: dict[str, Any] | None = None
        self._chat_history: list[Any] = []

    # ── Task ──────────────────────────────────────────────────────────────────

    def set_task(self, task: str | HumanMessage) -> None:
        """Store the initial human task shared across all agents."""
        if isinstance(task, HumanMessage):
            self._task = task
        else:
            self._task = HumanTaskMessage(content=str(task))

        self._task = apply_message_label(self._task, "Initial Human Task")

    def get_task(self) -> HumanMessage | None:
        return self._task

    # ── Plan ──────────────────────────────────────────────────────────────────

    def set_plan(self, plan: str | HumanMessage) -> None:
        """Store/overwrite the planner's current plan."""
        if plan is None:
            self._plan = None
            return
        if isinstance(plan, HumanMessage):
            self._plan = plan
        else:
            self._plan = HumanTaskMessage(content=str(plan))

        self._plan = apply_message_label(
            self._plan, "Plan for solving given Task")

    def get_plan(self) -> HumanMessage | None:
        return self._plan

    # ── Task formulation ──────────────────────────────────────────────────────

    def set_subtask_formulation(self, formulation: str | HumanMessage) -> None:
        """Store the planner's sub-task instruction for the actor."""
        if isinstance(formulation, HumanMessage):
            self._subtask_formulation = formulation
        else:
            self._subtask_formulation = TaskFormulationMessage(
                content=str(formulation))

    def get_subtask_formulation(self) -> HumanMessage | None:
        return self._subtask_formulation

    # ── Pending tool call ─────────────────────────────────────────────────────

    def set_pending_tool_call(self, tool_call: dict[str, Any]) -> None:
        """Store the actor's current tool call awaiting critic review."""
        self._pending_tool_call = tool_call

    def get_pending_tool_call(self) -> dict[str, Any] | None:
        return self._pending_tool_call

    # ── Chat history ──────────────────────────────────────────────────────────

    def add(self, message: Any) -> None:
        """Append a message to the chat history (keeps all messages)."""
        self._chat_history.append(message)

    def add_tool_call_and_response(
        self,
        ai_message: AIMessage,
        tool_response: ToolMessage,
    ) -> None:
        """Record an approved tool call and its execution result as a pair.

        Stores the AIMessage (with tool_calls) followed by the ToolMessage
        (with the exec_result or critic feedback) in chat history.
        """
        self._chat_history.append(ai_message)
        self._chat_history.append(tool_response)

    # ── Message builders ──────────────────────────────────────────────────────

    def build_planner_messages(self) -> list[Any]:
        """planner_system_prompt + task + chat history (includes plan)"""
        messages: list[Any] = [self._planner_system_prompt]
        if self._task:
            messages.append(self._task)
        messages.extend(self._chat_history)
        return messages

    def build_actor_messages(self) -> list[Any]:
        """actor_system_prompt + task + chat history + subtask formulation

        The chat history now contains tool call / tool response pairs which
        the Actor LLM understands natively.
        """
        messages: list[Any] = [self._actor_system_prompt]
        if self._task:
            messages.append(self._task)
        messages.extend(self._chat_history)
        if self._subtask_formulation:
            messages.append(self._subtask_formulation)
        return messages

    def build_critic_messages(self) -> list[Any]:
        """critic_system_prompt only

        The critic receives the system prompt; command details and syntax
        check results are injected by CriticAgent._compose_critic_messages().
        """
        messages: list[Any] = [self._critic_system_prompt]
        return messages

    def snapshot_memory(self) -> dict[str, Any]:
        return {
            "task": self._task,
            "plan": self._plan,
            "subtask": self._subtask_formulation,
            "pending_tool_call": self._pending_tool_call,
            "memory": self._chat_history,
        }
