from collections import deque
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.terminal_bench_supplementary.pipeline_messages import (
    CriticFeedbackMessage,
    ExecutionRequestCandidateMessage,
    TaskFormulationMessage,
)


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
        self._plan: HumanMessage | None = None
        self._task_formulation: HumanMessage | None = None
        self._execution_request_candidate: HumanMessage | None = None
        self._critic_feedback: HumanMessage | None = None
        self._short_term: deque[Any] = deque(maxlen=short_term_window)

    # ── Plan ──────────────────────────────────────────────────────────────────

    def set_plan(self, plan: Any) -> None:
        """Store/overwrite the planner's current plan."""
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
        self._plan = HumanMessage(content="## Current plan:\n" + body)

    def get_plan(self) -> HumanMessage | None:
        return self._plan

    # ── Task formulation ──────────────────────────────────────────────────────

    def set_task_formulation(self, formulation: str | HumanMessage) -> None:
        """Store the planner's sub-task instruction for the actor."""
        if isinstance(formulation, HumanMessage):
            self._task_formulation = formulation
        else:
            self._task_formulation = TaskFormulationMessage(content=str(formulation))

    def get_task_formulation(self) -> HumanMessage | None:
        return self._task_formulation

    # ── Execution request candidate ───────────────────────────────────────────

    def set_execution_request_candidate(self, candidate: str | HumanMessage) -> None:
        """Store the actor's proposed execution request."""
        if isinstance(candidate, HumanMessage):
            self._execution_request_candidate = candidate
        else:
            self._execution_request_candidate = ExecutionRequestCandidateMessage(content=str(candidate))

    def get_execution_request_candidate(self) -> HumanMessage | None:
        return self._execution_request_candidate

    # ── Critic feedback ───────────────────────────────────────────────────────

    def set_critic_feedback(self, feedback: str | HumanMessage) -> None:
        """Store the critic's latest feedback on the execution request candidate."""
        if isinstance(feedback, HumanMessage):
            self._critic_feedback = feedback
        else:
            self._critic_feedback = CriticFeedbackMessage(content=str(feedback))

    def get_critic_feedback(self) -> HumanMessage | None:
        return self._critic_feedback

    # ── Short-term history ────────────────────────────────────────────────────

    def add(self, message: Any) -> None:
        """Append a message to the short-term rolling window."""
        self._short_term.append(message)

    # ── Message builders ──────────────────────────────────────────────────────

    def build_planner_messages(self) -> list[Any]:
        """planner_system_prompt + plan + short-term history"""
        messages: list[Any] = [self._planner_system_prompt]
        if self._plan:
            messages.append(self._plan)
        messages.extend(self._short_term)
        return messages

    def build_actor_messages(self) -> list[Any]:
        """actor_system_prompt + task_formulation + short-term history + critic_feedback"""
        messages: list[Any] = [self._actor_system_prompt]
        if self._task_formulation:
            messages.append(self._task_formulation)
        messages.extend(self._short_term)
        if self._critic_feedback:
            messages.append(self._critic_feedback)
        return messages

    def build_critic_messages(self) -> list[Any]:
        """critic_system_prompt + execution_request_candidate"""
        messages: list[Any] = [self._critic_system_prompt]
        if self._execution_request_candidate:
            messages.append(self._execution_request_candidate)
        return messages
