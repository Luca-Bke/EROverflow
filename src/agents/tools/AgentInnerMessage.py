
from typing import Any, Literal

from langchain_core.messages import HumanMessage


class AgentInnerMessage(HumanMessage):
    """Internal agent feedback message, not originating from a real user.

    Serializes to role "user" for OpenAI-compatible APIs while carrying a
    distinct type label for internal routing between agent components.
    """

    type: Literal["agent_inner_voice"] = "agent_inner_voice"

    def __init__(self, content: str | list[str | dict], **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)