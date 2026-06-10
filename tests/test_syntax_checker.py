import json
import pytest
from agents.terminal_bench import TerminalBenchAgent
from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception


@pytest.fixture
def agent():
    return TerminalBenchAgent()


# --- _check_command_syntax ---

def test_valid_command_passes(agent):
    agent._check_command_syntax("echo hello")

def test_multiline_command_passes(agent):
    agent._check_command_syntax("echo hello\necho world")

def test_invalid_syntax_raises(agent):
    with pytest.raises(terminal_bench_format_exception, match="invalid shell syntax"):
        agent._check_command_syntax("if then done")

def test_unclosed_quote_raises(agent):
    with pytest.raises(terminal_bench_format_exception):
        agent._check_command_syntax("echo 'hello")

def test_empty_command_in_exec_request_raises(agent):
    payload = json.dumps({"kind": "exec_request", "command": "", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception, match="missing a command"):
        agent.postprocess_response(payload, updater=None)

def test_missing_command_field_raises(agent):
    payload = json.dumps({"kind": "exec_request", "timeout": 30})
    with pytest.raises(terminal_bench_format_exception, match="missing a command"):
        agent.postprocess_response(payload, updater=None)


# --- postprocess_response ---

def test_valid_exec_request_passes(agent):
    payload = json.dumps({"kind": "exec_request", "command": "echo hello", "timeout": 30})
    result = agent.postprocess_response(payload, updater=None)
    assert result == payload

def test_invalid_command_in_exec_request_raises(agent):
    payload = json.dumps({"kind": "exec_request", "command": "if then done", "timeout": 30})
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
