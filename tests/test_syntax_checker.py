import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from agents.terminal_bench import TerminalBenchAgent
from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.response_format_checker import ResponseFormatChecker
from a2a.types import Message, Part, TextPart


@pytest.fixture
def agent():
    mock_client = MagicMock()
    mock_client.invoke_async = AsyncMock()
    mock_client.rate_limited = MagicMock(return_value=False)
    mock_client.retry_log = MagicMock(return_value=[])
    return TerminalBenchAgent(
        llm_client=mock_client,
        system_prompt="test-system-prompt",
        recon_cmd="echo recon",
    )


# --- ExecRequestChecker.check_command_syntax ---

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


def test_empty_command_in_exec_request_raises(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        agent.validate_response(payload)


def test_missing_command_field_raises(agent):
    payload = json.dumps({"kind": "exec_request", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        agent.validate_response(payload)


# --- validate_response ---

def test_valid_exec_request_passes(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hello", "timeout": 30})
    result = agent.validate_response(payload)
    assert result["kind"] == "exec_request"


def test_invalid_command_in_exec_request_raises(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "if then done", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception):
        agent.validate_response(payload)


def test_final_passes(agent):
    payload = json.dumps({"kind": "final"})
    result = agent.validate_response(payload)
    assert result["kind"] == "final"


def test_plan_passes(agent):
    payload = json.dumps({"kind": "plan", "steps": ["explore", "build", "verify"]})
    result = agent.validate_response(payload)
    assert result["kind"] == "plan"


def test_empty_plan_raises(agent):
    payload = json.dumps({"kind": "plan", "steps": []})
    with pytest.raises(terminal_bench_format_exception, match="non-empty 'steps'"):
        agent.validate_response(payload)


def test_invalid_json_raises(agent):
    with pytest.raises(terminal_bench_format_exception, match="not valid JSON"):
        agent.validate_response("not json")


def test_unknown_kind_raises(agent):
    payload = json.dumps({"kind": "unknown_thing"})
    with pytest.raises(terminal_bench_format_exception, match="unknown kind"):
        agent.validate_response(payload)


def test_invalid_timeout_raises(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": -1})
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        agent.validate_response(payload)


def test_zero_timeout_raises(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": 0})
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        agent.validate_response(payload)


# --- ResponseFormatChecker deterministic normalisation ---

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


# --- retry logic in handle_request_iteration ---

def _make_message(text: str) -> Message:
    return Message(
        kind="message",
        role="user",
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id="test-id",
    )


def _mock_checker(agent, approved: bool = True, feedback: str = "send one command"):
    """Replace the checker agent with a deterministic stub (no real LLM)."""
    from agents.checker_agent import CheckVerdict
    agent._checker_agent.review = AsyncMock(
        return_value=CheckVerdict(approved=approved, feedback=feedback))


async def test_retry_succeeds_on_second_attempt(agent):
    bad = "not valid json"
    good = json.dumps(
        {"kind": "exec_request", "command": "echo hello", "timeout": 30})

    call_count = 0

    async def fake_invoke(messages):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        mock.content = bad if call_count == 1 else good
        return mock

    agent._llm_client.invoke_async = fake_invoke
    _mock_checker(agent, approved=True)

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    result = await agent.handle_request_iteration(_make_message(exec_payload), updater)

    assert result == good
    assert call_count == 2


async def test_retry_fails_after_max_attempts(agent):
    async def fake_invoke(messages):
        mock = MagicMock()
        mock.content = "not valid json"
        return mock

    agent._llm_client.invoke_async = fake_invoke
    _mock_checker(agent, approved=False)

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    with pytest.raises(terminal_bench_format_exception):
        await agent.handle_request_iteration(_make_message(exec_payload), updater)
