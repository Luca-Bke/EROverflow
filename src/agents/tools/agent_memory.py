from collections import deque
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.terminal_bench_supplementary.pipeline_messages import (
    CriticFeedbackMessage,
    ExecutionRequestCandidateMessage,
    HumanTaskMessage,
    TaskFormulationMessage,
)
from agents.terminal_bench_supplementary.utils import apply_message_label


class AgentMemory:
    """Shared memory for a planner-actor-critic agent pipeline.

    Holds the three agent system prompts and the shared state they read/write:
      - plan: planner's current execution plan (one message, overwritten on update)
      - task_formulation: planner's sub-task instruction for the actor (one message)
      - short_term: rolling window of execution results and AI responses
      - execution_request_candidate: actor's proposed shell command (one message)
      - critic_feedback: critic's latest verdict on the candidate (one message)

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
        self._execution_request_candidate: HumanMessage | None = None
        self._critic_feedback: HumanMessage | None = None
        self._short_term: deque[Any] = deque(maxlen=short_term_window)

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
            self._plan, "Plan Created By Planner Agent")

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

        self._subtask_formulation = apply_message_label(
            self._subtask_formulation, "Subtask Created By Planner Agent")

    def get_subtask_formulation(self) -> HumanMessage | None:
        return self._subtask_formulation

    # ── Execution request candidate ───────────────────────────────────────────

    def set_execution_request_candidate(self, candidate: str | HumanMessage) -> None:
        """Store the actor's proposed execution request."""
        if isinstance(candidate, HumanMessage):
            self._execution_request_candidate = candidate
        else:
            self._execution_request_candidate = ExecutionRequestCandidateMessage(
                content=str(candidate))

    def get_execution_request_candidate(self) -> HumanMessage | None:
        return self._execution_request_candidate

    # ── Critic feedback ───────────────────────────────────────────────────────

    def set_critic_feedback(self, feedback: str | HumanMessage) -> None:
        """Store the critic's latest feedback on the execution request candidate."""
        if isinstance(feedback, HumanMessage):
            self._critic_feedback = feedback
        else:
            self._critic_feedback = CriticFeedbackMessage(
                content=str(feedback))

    def get_critic_feedback(self) -> HumanMessage | None:
        return self._critic_feedback

    # ── Short-term history ────────────────────────────────────────────────────

    def add(self, message: Any) -> None:
        """Append a message to the short-term rolling window."""
        self._short_term.append(message)

    # ── Message builders ──────────────────────────────────────────────────────

    def build_planner_messages(self) -> list[Any]:
        """planner_system_prompt + task + plan + short-term history"""
        messages: list[Any] = [self._planner_system_prompt]
        if self._task:
            messages.append(self._task)
        if self._plan:
            messages.append(self._plan)
        messages.append(HumanMessage(content="The following is the short time history of executed commands:"))
        messages.extend(self._short_term)
        return messages

    def build_actor_messages(self) -> list[Any]:
        """actor_system_prompt + task + task_formulation + short-term history + critic_feedback"""
        messages: list[Any] = [self._actor_system_prompt]
        if self._task:
            messages.append(self._task)
        if self._subtask_formulation:
            messages.append(self._subtask_formulation)
        messages.append(HumanMessage(content="The following is the short time history of executed commands:"))
        messages.extend(self._short_term)
        if self._critic_feedback:
            messages.append(self._critic_feedback)
        return messages

    def build_critic_messages(self) -> list[Any]:
        """critic_system_prompt + task + execution_request_candidate"""
        messages: list[Any] = [self._critic_system_prompt]
        if self._task:
            messages.append(self._task)
        if self._subtask_formulation:
            messages.append(self._subtask_formulation)
        if self._execution_request_candidate:
            messages.append(self._execution_request_candidate)
        return messages
