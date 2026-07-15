"""Shared retryable-error classification for LLM clients.

Distinguishes a *failed* attempt from "still computing" purely by outcome:
  - a returned response (HTTP 200) → done, use it
  - 5xx / timeout / connection error → this attempt failed → retryable
  - other 4xx (auth, bad request) → not retryable → give up immediately

An open request counts as "still computing" until the client's request timeout
fires (configured on the ChatOpenAI instance, not here).

The actual retry logic lives in the agent layer (Planner, Actor, Critic) so
that each attempt shows up as a separate trace in LangSmith.
"""

from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

# HTTP status codes worth retrying (server-side / transient).
RETRYABLE_STATUS = {500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    """Return True if the exception represents a transient, retryable failure."""
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in RETRYABLE_STATUS
    return False


async def ainvoke_with_backoff(
    llm: Any,
    messages: list[Any],
    *,
    max_retries: int,
    base_delay: float,
    retry_log: list[dict],
    tools: list[dict] | None = None,
    response_format: dict | None = None,
) -> Any:
    """Invoke ``llm.ainvoke(messages)`` once (no retry loop here).

    Retry logic is handled by the caller (agent layer) so each attempt
    appears as a separate trace in LangSmith. This function performs a
    single invocation and lets errors propagate.

    ``tools`` and ``response_format`` are forwarded to ``llm.ainvoke()`` as
    keyword arguments when provided (for tool-calling and structured output).
    """
    invoke_kwargs: dict[str, Any] = {}
    if tools is not None:
        invoke_kwargs["tools"] = tools
    if response_format is not None:
        invoke_kwargs["response_format"] = response_format

    if invoke_kwargs:
        return await llm.ainvoke(messages, **invoke_kwargs)
    return await llm.ainvoke(messages)
