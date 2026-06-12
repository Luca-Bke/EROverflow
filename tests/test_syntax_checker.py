import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from agents.terminal_bench import TerminalBenchAgent
from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception
from agents.tools.bash_syntax_checker import ExecRequestChecker
from a2a.types import Message, Part, TextPart


@pytest.fixture
def agent():
    return TerminalBenchAgent()


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
        agent.postprocess_response(payload, updater=None)


def test_missing_command_field_raises(agent):
    payload = json.dumps({"kind": "exec_request", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception, match="empty command"):
        agent.postprocess_response(payload, updater=None)


# --- postprocess_response ---

def test_valid_exec_request_passes(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hello", "timeout": 30})
    result = agent.postprocess_response(payload, updater=None)
    assert result == payload


def test_invalid_command_in_exec_request_raises(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "if then done", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception):
        agent.postprocess_response(payload, updater=None)


def test_final_passes(agent):
    payload = json.dumps({"kind": "final"})
    result = agent.postprocess_response(payload, updater=None)
    assert result == payload


def test_invalid_json_raises(agent):
    with pytest.raises(terminal_bench_format_exception, match="not valid JSON"):
        agent.postprocess_response("not json", updater=None)


def test_unknown_kind_raises(agent):
    payload = json.dumps({"kind": "unknown_thing"})
    with pytest.raises(terminal_bench_format_exception, match="unknown kind"):
        agent.postprocess_response(payload, updater=None)


def test_invalid_timeout_raises(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": -1})
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        agent.postprocess_response(payload, updater=None)


def test_zero_timeout_raises(agent):
    payload = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": 0})
    with pytest.raises(terminal_bench_format_exception, match="invalid timeout"):
        agent.postprocess_response(payload, updater=None)


# --- retry logic in handle_request_iteration ---

def _make_message(text: str) -> Message:
    return Message(
        kind="message",
        role="user",
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id="test-id",
    )


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

    agent._invoke_llm_async = fake_invoke

    task_payload = json.dumps({"kind": "task", "instruction": "do something"})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    result = await agent.handle_request_iteration(_make_message(task_payload), updater)

    assert result == good
    assert call_count == 2


async def test_retry_fails_after_max_attempts(agent):
    async def fake_invoke(messages):
        mock = MagicMock()
        mock.content = "not valid json"
        return mock

    agent._invoke_llm_async = fake_invoke

    task_payload = json.dumps({"kind": "task", "instruction": "do something"})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    with pytest.raises(terminal_bench_format_exception):
        await agent.handle_request_iteration(_make_message(task_payload), updater)
