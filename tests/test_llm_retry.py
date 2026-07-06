from unittest.mock import AsyncMock, patch

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


# ── ainvoke_with_backoff (asyncio.sleep patched → no real waiting) ──────────────

async def test_succeeds_after_one_retry():
    ok = object()
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=[_Status(504), ok])
    log: list[dict] = []

    with patch("agents.llm_clients.retry.asyncio.sleep", new=AsyncMock()):
        result = await ainvoke_with_backoff(
            llm, [], max_retries=4, base_delay=5, retry_log=log)

    assert result is ok
    assert llm.ainvoke.await_count == 2
    assert log and log[0]["delay_seconds"] == 5


async def test_exhausts_then_raises():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=_Status(504))
    log: list[dict] = []

    with patch("agents.llm_clients.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(APIStatusError):
            await ainvoke_with_backoff(
                llm, [], max_retries=3, base_delay=5, retry_log=log)

    assert llm.ainvoke.await_count == 3
    assert any(e.get("exhausted") for e in log)


async def test_non_retryable_is_not_retried():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=_Status(400))
    log: list[dict] = []

    with patch("agents.llm_clients.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(APIStatusError):
            await ainvoke_with_backoff(
                llm, [], max_retries=4, base_delay=5, retry_log=log)

    assert llm.ainvoke.await_count == 1
