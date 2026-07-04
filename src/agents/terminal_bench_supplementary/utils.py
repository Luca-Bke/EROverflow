import inspect
import json
import time
from functools import wraps

from langchain_core.messages import BaseMessage
from langsmith import traceable

from agents.configuration import config


# ── Response classification ───────────────────────────────────────────────────

def is_final_response(response_result: str) -> bool:
    """Return True if the response JSON carries kind=final."""
    if not response_result:
        return False
    try:
        return json.loads(response_result).get("kind") == "final"
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        return False


# ── LangSmith session tracing ─────────────────────────────────────────────────

@traceable(name="agent_session_trace", run_type="chain")
def emit_session_trace(
    history: list[dict],
    turn_count: int,
    rate_limited: bool,
    retry_log: list[dict],
) -> dict:
    """Emit a single LangSmith trace summarising one completed agent session."""
    return {
        "turn_count": turn_count,
        "rate_limited": rate_limited,
        "retry_count": len(retry_log),
        "history_length": len(history),
        "completed": not rate_limited,
    }


@traceable(name="timer_trace", run_type="chain")
def emit_timer_trace(timer_sessions: list[list[dict]]) -> dict:
    """Emit a single LangSmith trace summarising one completed agent session."""
    return {
        "timer_sessions": timer_sessions,
    }


# ── Execution timing ──────────────────────────────────────────────────────────


class TimeTracer:

    timer_sessions: list[list[dict]] = []
    current_session: list[dict] = []

    last_end_time = {}

    def __init__():
        TimeTracer.timer_sessions = []
        TimeTracer.current_session = []

    def new_session():
        TimeTracer.timer_sessions.append(TimeTracer.current_session)
        TimeTracer.current_session = []

    def timed(entry_name: str):
        def decorator(func):
            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        TimeTracer.current_session.append(
                            {f"[{entry_name}]": time.perf_counter() - start})
                return async_wrapper

            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    TimeTracer.current_session.append(
                        {f"[{entry_name}]": time.perf_counter() - start})
            return wrapper
        return decorator

    def inverse_timed(entry_name: str):
        def decorator(func):
            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start = time.perf_counter()
                    last_end = TimeTracer.last_end_time.get(entry_name)
                    if last_end is not None:
                        TimeTracer.current_session.append(
                            {f"[{entry_name}]": start - last_end})
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        TimeTracer.last_end_time[entry_name] = time.perf_counter(
                        )
                return async_wrapper

            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                last_end = TimeTracer.last_end_time.get(entry_name)
                if last_end is not None:
                    TimeTracer.current_session.append(
                        {f"[{entry_name}]": start - last_end})
                try:
                    return func(*args, **kwargs)
                finally:
                    TimeTracer.last_end_time[entry_name] = time.perf_counter()
            return wrapper
        return decorator


# ── Output truncation ─────────────────────────────────────────────────────────

def truncate_field(value: str, budget: int = config.MAX_OUTPUT_CHARS) -> str:
    """Keep the head and tail of a long string, eliding the middle."""
    if len(value) <= budget:
        return value
    half = budget // 2
    elided = len(value) - 2 * half
    return f"{value[:half]}\n…[{elided} chars truncated]…\n{value[-half:]}"


def truncate_exec_result(input_text: str) -> str:
    """Truncate stdout/stderr inside an exec_result JSON before it enters memory.

    Long commands (apt-get, pip, training runs) can emit tens of thousands of
    lines. Storing them verbatim blows the rolling context window and can crash
    the A2A gateway. Head+tail of each stream is kept instead.
    """
    try:
        data = json.loads(input_text)
    except (json.JSONDecodeError, ValueError):
        return truncate_field(input_text)

    for field in ("stdout", "stderr", "output"):
        if isinstance(data.get(field), str):
            data[field] = truncate_field(data[field])
    return json.dumps(data)


def apply_message_label(message: BaseMessage, label: str) -> BaseMessage:
    if not message.content.startswith(f"[{label}]"):
        return type(message)(content=f"[{label}]\n{message.content}")
    else:
        return message
