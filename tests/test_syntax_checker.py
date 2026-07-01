import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import SystemMessage

from agents.terminal_bench import TerminalBenchAgent
from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.response_format_checker import ResponseFormatChecker
from agents.critic import CriticAgent, CriticVerdict
from agents.planner import PlannerOutput
from a2a.types import Message, Part, TextPart


@pytest.fixture
def agent():
    """A TerminalBenchAgent whose sub-agents are stubbed per test — no LLM."""
    client = MagicMock()
    client.invoke_async = AsyncMock()
    client.rate_limited = MagicMock(return_value=False)
    client.retry_log = MagicMock(return_value=[])
    return TerminalBenchAgent(
        llm_client=client,
        planner_system_prompt=SystemMessage(content="planner"),
        actor_system_prompt=SystemMessage(content="actor"),
        critic_system_prompt=SystemMessage(content="critic"),
        max_critic_actor_rounds=10,
        short_term_window=10,
    )


def _make_message(text: str) -> Message:
    return Message(
        kind="message",
        role="user",
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id="test-id",
    )


def _ai(content: str):
    """A stand-in for an actor result message carrying `.content`."""
    return MagicMock(content=content)


# --- ExecRequestChecker.check_command_syntax (no LLM) --------------------------

def test_valid_command_passes():
    ExecRequestChecker.check_command_syntax("echo hello")


def test_multiline_command_passes():
    ExecRequestChecker.check_command_syntax("echo hello\necho world")


def test_invalid_syntax_raises():
    with pytest.raises(terminal_bench_format_exception, match="invalid shell syntax"):
        ExecRequestChecker.check_command_syntax("if then done")


def test_unclosed_quote_raises():
    with pytest.raises(terminal_bench_format_exception):
        ExecRequestChecker.check_command_syntax("echo 'hello")


def test_empty_command_raises():
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        ExecRequestChecker.check_command_syntax("")


def test_interactive_vim_raises():
    with pytest.raises(terminal_bench_format_exception, match="Interactive command"):
        ExecRequestChecker.check_command_syntax("vim file.txt")


def test_interactive_less_raises():
    with pytest.raises(terminal_bench_format_exception, match="Interactive command"):
        ExecRequestChecker.check_command_syntax("less output.log")


def test_interactive_python_repl_raises():
    with pytest.raises(terminal_bench_format_exception, match="REPL"):
        ExecRequestChecker.check_command_syntax("python")


def test_python_with_c_flag_passes():
    ExecRequestChecker.check_command_syntax("python -c 'print(1)'")


def test_destructive_rm_rf_root_raises():
    with pytest.raises(terminal_bench_format_exception, match="destructive"):
        ExecRequestChecker.check_command_syntax("rm -rf /")


def test_fork_bomb_raises():
    with pytest.raises(terminal_bench_format_exception, match="destructive"):
        ExecRequestChecker.check_command_syntax(":(){ :|:& };:")


# --- CriticAgent._validate_response (static, no LLM) --------------------------

def test_empty_command_in_exec_request_raises():
    payload = json.dumps(
        {"kind": "exec_request", "command": "", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        CriticAgent._validate_response(payload)


def test_missing_command_field_raises():
    payload = json.dumps({"kind": "exec_request", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        CriticAgent._validate_response(payload)


def test_valid_exec_request_passes():
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hello", "timeout": 30})
    result = CriticAgent._validate_response(payload)
    assert result["kind"] == "exec_request"


def test_invalid_command_in_exec_request_raises():
    payload = json.dumps(
        {"kind": "exec_request", "command": "if then done", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception):
        CriticAgent._validate_response(payload)


def test_final_passes():
    payload = json.dumps({"kind": "final"})
    result = CriticAgent._validate_response(payload)
    assert result["kind"] == "final"


def test_invalid_json_raises():
    with pytest.raises(terminal_bench_format_exception, match="not valid JSON"):
        CriticAgent._validate_response("not json")


def test_unknown_kind_raises():
    payload = json.dumps({"kind": "unknown_thing"})
    with pytest.raises(terminal_bench_format_exception, match="unknown kind"):
        CriticAgent._validate_response(payload)


def test_invalid_timeout_raises():
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": -1})
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        CriticAgent._validate_response(payload)


def test_zero_timeout_raises():
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": 0})
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        CriticAgent._validate_response(payload)


# --- ResponseFormatChecker deterministic normalisation (no LLM) ---------------

def test_clean_json_passes_through():
    payload = json.dumps({"kind": "exec_request", "command": "ls", "timeout": 30})
    assert ResponseFormatChecker.check_agent_response_valid_json(payload)["command"] == "ls"


def test_think_wrapped_json_is_stripped():
    raw = '<think>let me plan this carefully</think>\n{"kind": "final"}'
    result = ResponseFormatChecker.check_agent_response_valid_json(raw)
    assert result["kind"] == "final"


def test_multiline_think_then_exec_request():
    raw = ('<think>\nstep 1\nstep 2\n</think>'
           '{"kind": "exec_request", "command": "echo hi", "timeout": 30}')
    result = ResponseFormatChecker.check_agent_response_valid_json(raw)
    assert result["kind"] == "exec_request"


def test_multiple_objects_raise_send_first():
    raw = ('{"kind": "exec_request", "command": "echo a", "timeout": 30}\n'
           '{"kind": "exec_request", "command": "echo b", "timeout": 30}')
    with pytest.raises(terminal_bench_format_exception, match="Multiple JSON objects"):
        ResponseFormatChecker.check_agent_response_valid_json(raw)


def test_garbage_raises_not_valid_json():
    with pytest.raises(terminal_bench_format_exception, match="not valid JSON"):
        ResponseFormatChecker.check_agent_response_valid_json("totally not json")


# --- critic-loop retry semantics in handle_request_iteration ------------------

async def test_retry_succeeds_on_second_attempt(agent):
    bad = "not valid json"
    good = json.dumps(
        {"kind": "exec_request", "command": "echo hello", "timeout": 30})

    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=["step"], task_formulation="do x"))
    agent._actor_agent.invoke = AsyncMock(side_effect=[_ai(bad), _ai(good)])
    agent._critic_agent.invoke = AsyncMock(side_effect=[
        CriticVerdict(approved=False, feedback="send valid JSON"),
        CriticVerdict(approved=True, feedback=""),
    ])

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock())

    assert result == good
    assert agent._actor_agent.invoke.await_count == 2


async def test_retry_fails_after_max_attempts(agent):
    bad = "not valid json"

    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=["step"], task_formulation="do x"))
    agent._actor_agent.invoke = AsyncMock(return_value=_ai(bad))
    agent._critic_agent.invoke = AsyncMock(
        return_value=CriticVerdict(approved=False, feedback="still wrong"))

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock())

    # Critic never approves → loop exhausts and a valid final is returned
    # (never None, so the caller can't crash on a None payload).
    assert json.loads(result)["kind"] == "final"
    assert agent._actor_agent.invoke.await_count == agent._max_critic_actor_rounds
