"""Shared exponential-backoff retry for LLM clients.

Distinguishes a *failed* attempt from "still computing" purely by outcome:
  - a returned response (HTTP 200) → done, use it
  - 5xx / timeout / connection error → this attempt failed → retry with backoff
  - other 4xx (auth, bad request) → not retryable → give up immediately

An open request counts as "still computing" until the client's request timeout
fires (configured on the ChatOpenAI instance, not here).
"""

import asyncio
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
) -> Any:
    """Invoke ``llm.ainvoke(messages)`` with exponential backoff on transient errors.

    Retries on 429 / 5xx / timeout / connection errors up to ``max_retries``
    attempts, waiting ``base_delay * 2**attempt`` seconds between tries. Re-raises
    on a non-retryable error or once attempts are exhausted; ``retry_log`` is
    appended to for observability.
    """
    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 — classified below
            last = attempt == attempts - 1
            if not is_retryable(exc) or last:
                retry_log.append({
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "exhausted": True,
                    "retryable": is_retryable(exc),
                    "error": str(exc)[:300],
                })
                raise
            delay = base_delay * (2 ** attempt)
            retry_log.append({
                "attempt": attempt + 1,
                "max_attempts": attempts,
                "delay_seconds": delay,
                "error": str(exc)[:300],
            })
            print(
                f"LLM call failed (attempt {attempt + 1}/{attempts}: "
                f"{type(exc).__name__}), retrying in {delay:.0f}s..."
            )
            await asyncio.sleep(delay)
