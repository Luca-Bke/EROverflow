import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import SystemMessage

from agents.terminal_bench import TerminalBenchAgent
from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.response_format_checker import ResponseFormatChecker
from agents.critic import CriticVerdict
from agents.planner import PlannerOutput
from a2a.types import Message, Part, TextPart


@pytest.fixture
def agent():
    """A TerminalBenchAgent whose sub-agents are stubbed per test — no LLM."""
    client = MagicMock()
    client.invoke_async = AsyncMock()
    client.invoke_with_tools_async = AsyncMock()
    client.invoke_with_response_format_async = AsyncMock()
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


def _ai_with_tool_call(command: str, timeout: int = 300):
    """An AIMessage carrying a tool_calls list."""
    from langchain_core.messages import AIMessage
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_test",
                "name": "execute_command",
                "args": {"command": command, "timeout": timeout},
            }
        ],
    )


# --- ExecRequestChecker.check_command_syntax (no LLM) --------------------------

def test_valid_command_passes():
    ExecRequestChecker.check_command_syntax("echo hello")


def test_multiline_command_passes():
    ExecRequestChecker.check_command_syntax("echo hello\necho world")


def test_invalid_syntax_raises(monkeypatch):
    monkeypatch.setattr(ExecRequestChecker, '_bash_available', True)
    with pytest.raises(terminal_bench_format_exception, match="invalid shell syntax"):
        ExecRequestChecker.check_command_syntax("if then done")


def test_unclosed_quote_raises(monkeypatch):
    monkeypatch.setattr(ExecRequestChecker, '_bash_available', True)
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


# --- ExecRequestChecker validation (static, no LLM) --------------------------
# The Critic no longer parses JSON — tool schema guarantees structure.
# These tests verify ExecRequestChecker directly (used by Critic internally).

def test_empty_command_in_exec_request_raises():
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        ExecRequestChecker.check_command_syntax("")


def test_missing_command_field_raises():
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        ExecRequestChecker.check_exec_request({"timeout": 30})


def test_valid_exec_request_passes():
    req = {"command": "echo hello", "timeout": 30}
    ExecRequestChecker.check_exec_request(req)  # should not raise


def test_invalid_command_in_exec_request_raises(monkeypatch):
    monkeypatch.setattr(ExecRequestChecker, '_bash_available', True)
    with pytest.raises(terminal_bench_format_exception):
        ExecRequestChecker.check_command_syntax("if then done")


def test_final_passes():
    """finalize no longer goes through Critic — it's a plain text Actor
    response (no tool call). Kept as no-op to document behaviour."""
    pass


def test_invalid_json_raises():
    """JSON parsing is no longer done by the Critic — tool schema guarantees
    valid arguments. Kept as no-op."""
    pass


def test_unknown_kind_raises():
    """kind field is no longer checked — tool calls have a fixed name.
    Kept as no-op."""
    pass


def test_invalid_timeout_raises():
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        ExecRequestChecker.check_exec_request(
            {"command": "echo hi", "timeout": -1})


def test_zero_timeout_raises():
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        ExecRequestChecker.check_exec_request(
            {"command": "echo hi", "timeout": 0})


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
    good_cmd = "echo hello"

    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=["step"], task_formulation="do x"))
    agent._actor_agent.invoke = AsyncMock(side_effect=[
        _ai_with_tool_call(good_cmd),
        _ai_with_tool_call(good_cmd),
    ])
    agent._critic_agent.invoke = AsyncMock(side_effect=[
        CriticVerdict(approved=False, feedback="send valid JSON"),
        CriticVerdict(approved=True, feedback=""),
    ])

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock())

    result_dict = json.loads(result)
    assert result_dict["kind"] == "exec_request"
    assert result_dict["command"] == good_cmd
    assert agent._actor_agent.invoke.await_count == 2


async def test_retry_fails_after_max_attempts(agent):
    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=["step"], task_formulation="do x"))
    agent._actor_agent.invoke = AsyncMock(
        return_value=_ai_with_tool_call("bad cmd"))
    agent._critic_agent.invoke = AsyncMock(
        return_value=CriticVerdict(approved=False, feedback="still wrong"))

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock())

    # Critic never approves → loop exhausts and returns a no-op exec_request
    # to continue in the next turn
    result_dict = json.loads(result)
    assert result_dict["kind"] == "exec_request"
    assert result_dict["command"] == "true"
    assert agent._actor_agent.invoke.await_count == agent._max_critic_actor_rounds
