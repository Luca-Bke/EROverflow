from unittest.mock import AsyncMock

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from agents.llm_clients.retry import ainvoke_with_backoff, is_retryable


# ── Lightweight fakes: skip the heavy openai __init__, keep isinstance + status ──

class _Status(APIStatusError):
    def __init__(self, code: int) -> None:
        self.status_code = code


class _RateLimit(RateLimitError):
    def __init__(self) -> None:
        self.status_code = 429


class _Timeout(APITimeoutError):
    def __init__(self) -> None:
        pass


class _Conn(APIConnectionError):
    def __init__(self) -> None:
        pass


# ── is_retryable ────────────────────────────────────────────────────────────

def test_is_retryable_5xx():
    assert is_retryable(_Status(500)) is True
    assert is_retryable(_Status(502)) is True
    assert is_retryable(_Status(503)) is True
    assert is_retryable(_Status(504)) is True


def test_not_retryable_4xx():
    assert is_retryable(_Status(400)) is False
    assert is_retryable(_Status(401)) is False
    assert is_retryable(_Status(404)) is False


def test_retryable_timeout_conn_ratelimit():
    assert is_retryable(_Timeout()) is True
    assert is_retryable(_Conn()) is True
    assert is_retryable(_RateLimit()) is True


def test_not_retryable_generic_error():
    assert is_retryable(ValueError("nope")) is False


# ── ainvoke_with_backoff (single invocation, no retry loop) ─────────────────

async def test_single_invoke_success():
    ok = object()
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=ok)
    log: list[dict] = []

    result = await ainvoke_with_backoff(
        llm, [], max_retries=4, base_delay=5, retry_log=log)

    assert result is ok
    assert llm.ainvoke.await_count == 1
    assert log == []  # no retries logged


async def test_single_invoke_failure_propagates():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=_Status(504))
    log: list[dict] = []

    with pytest.raises(APIStatusError):
        await ainvoke_with_backoff(
            llm, [], max_retries=4, base_delay=5, retry_log=log)

    assert llm.ainvoke.await_count == 1  # only one attempt
    assert log == []  # no retry logic in ainvoke_with_backoff anymore


async def test_non_retryable_failure_propagates():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=_Status(400))
    log: list[dict] = []

    with pytest.raises(APIStatusError):
        await ainvoke_with_backoff(
            llm, [], max_retries=4, base_delay=5, retry_log=log)

    assert llm.ainvoke.await_count == 1
