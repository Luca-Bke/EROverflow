import json

from langsmith import traceable

from agents.configuration import config


# ── Response classification ───────────────────────────────────────────────────

def is_final_response(response_result: str) -> bool:
    """Return True if the response JSON carries kind=final."""
    try:
        return json.loads(response_result).get("kind") == "final"
    except (json.JSONDecodeError, AttributeError, ValueError):
        return False


# ── LangSmith session tracing ─────────────────────────────────────────────────

@traceable(name="agent_session", run_type="chain")
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
