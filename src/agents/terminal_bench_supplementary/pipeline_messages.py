from typing import Any, Literal

from langchain_core.messages import HumanMessage, ToolMessage


class AgentInnerMessage(HumanMessage):
    """Internal agent feedback message, not originating from a real user.

    Serializes to role "user" for OpenAI-compatible APIs while carrying a
    distinct type label for internal routing between agent components.
    """

    type: Literal["agent_inner_voice"] = "agent_inner_voice"

    def __init__(self, content: str | list[str | dict], **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)


class ExecutionResultMessage(HumanMessage):
    """Shell execution result returned from the environment."""

    type: Literal["execution_result"] = "execution_result"

    def __init__(self, content: str | list[str | dict], **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)


class CriticFeedbackMessage(HumanMessage):
    """Feedback from the critic agent on an execution request candidate."""

    type: Literal["critic_feedback"] = "critic_feedback"

    def __init__(self, content: str | list[str | dict], **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)


class TaskFormulationMessage(HumanMessage):
    """Sub-task instruction produced by the planner agent for the actor."""

    type: Literal["task_formulation"] = "task_formulation"

    def __init__(self, content: str | list[str | dict], **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)


class ToolResponseMessage(ToolMessage):
    """Response to a tool call — either execution result or critic feedback.

    Carries a ``tool_call_id`` linking it back to the originating tool call.
    """

    type: Literal["tool"] = "tool"

    def __init__(
        self,
        content: str | list[str | dict],
        tool_call_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(content=content, tool_call_id=tool_call_id, **kwargs)


class HumanTaskMessage(HumanMessage):
    """The initial task issued by the human that all agents must complete."""

    type: Literal["human_task"] = "human_task"

    def __init__(self, content: str | list[str | dict], **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)
