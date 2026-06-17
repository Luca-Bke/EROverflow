import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.terminal_bench import TerminalBenchAgent, RECON_CMD, MAX_OUTPUT_CHARS
from agents.checker_agent import CheckerAgent, CheckVerdict
from agents.tools.agent_memory import AgentMemory
from a2a.types import Message, Part, TextPart


@pytest.fixture
def agent():
    return TerminalBenchAgent()


def _make_message(text: str) -> Message:
    return Message(
        kind="message",
        role="user",
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id="test-id",
    )


def _mock_checker(agent, approved=True, feedback="send one command"):
    agent._checker_agent.review = AsyncMock(
        return_value=CheckVerdict(approved=approved, feedback=feedback))


# ── CheckerAgent verdict parsing (no LLM) ──────────────────────────────────────

def test_parse_verdict_approved():
    v = CheckerAgent._parse_verdict('{"approved": true, "feedback": ""}', None)
    assert v.approved is True


def test_parse_verdict_rejected_with_feedback():
    raw = '{"approved": false, "feedback": "Send only the first command."}'
    v = CheckerAgent._parse_verdict(raw, None)
    assert v.approved is False
    assert "first command" in v.feedback


def test_parse_verdict_embedded_in_prose():
    raw = 'Here is my verdict:\n{"approved": false, "feedback": "fix it"}\nthanks'
    v = CheckerAgent._parse_verdict(raw, None)
    assert v.approved is False
    assert v.feedback == "fix it"


def test_parse_verdict_unparseable_does_not_approve():
    v = CheckerAgent._parse_verdict("totally not json", "orig error")
    assert v.approved is False


async def test_checker_review_falls_open_on_llm_error():
    """When the underlying LLM is unavailable, review() returns error=True."""
    checker = CheckerAgent()
    checker._create_llm = MagicMock(side_effect=RuntimeError("no api key"))
    v = await checker.review('{"kind":"final"}', None)
    assert v.error is True
    assert v.approved is False


# ── AgentMemory: plan slot + rolling window ────────────────────────────────────

def test_memory_set_plan_from_dict():
    mem = AgentMemory(SystemMessage(content="sys"), short_term_window=10)
    mem.set_plan({"kind": "plan", "steps": ["a", "b", "c"]})
    plan = mem.get_plan()
    assert plan is not None
    assert "1. a" in plan.content and "3. c" in plan.content


def test_memory_window_keeps_last_5_pairs():
    mem = AgentMemory(SystemMessage(content="sys"), short_term_window=10)
    mem.set_task(HumanMessage(content="task"))
    mem.set_plan({"steps": ["do x"]})
    # Add 8 exchange pairs = 16 messages; only last 10 short-term should remain.
    for i in range(8):
        mem.add(HumanMessage(content=f"exec_result {i}"))
        mem.add(AIMessage(content=f"response {i}"))

    messages = mem.build_messages()
    # system + task + plan + 10 short-term
    assert messages[0].content == "sys"
    assert messages[1].content == "task"
    assert "do x" in messages[2].content
    short_term = messages[3:]
    assert len(short_term) == 10
    # Oldest surviving pair is exchange #3 (pairs 0,1,2 rolled off)
    assert short_term[0].content == "exec_result 3"
    assert short_term[-1].content == "response 7"


# ── Plan turn handling in the loop ─────────────────────────────────────────────

async def test_plan_turn_is_internal_then_exec_sent(agent):
    plan = json.dumps({"kind": "plan", "steps": ["explore", "build"]})
    exec_req = json.dumps(
        {"kind": "exec_request", "command": "ls -la", "timeout": 300})

    call_count = 0

    async def fake_invoke(messages):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        mock.content = plan if call_count == 1 else exec_req
        return mock

    agent._invoke_llm_async = fake_invoke
    _mock_checker(agent, approved=True)

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    result = await agent.handle_request_iteration(_make_message(exec_payload), updater)

    # The plan is internal: it must NOT be the response sent to green.
    assert result == exec_req
    assert call_count == 2
    # Plan was stored in long-term memory.
    stored = agent._memory.get_plan()
    assert stored is not None
    assert "explore" in stored.content


async def test_plan_turn_budget_does_not_exhaust_syntax_retries(agent):
    """Repeated plan turns must not count against the syntax retry budget."""
    plan = json.dumps({"kind": "plan", "steps": ["x"]})
    exec_req = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": 300})

    call_count = 0

    async def fake_invoke(messages):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        # 4 plan turns (> _max_syntax_retries would matter if counted), then exec
        mock.content = plan if call_count <= 4 else exec_req
        return mock

    agent._invoke_llm_async = fake_invoke
    _mock_checker(agent, approved=True)

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    result = await agent.handle_request_iteration(_make_message(exec_payload), updater)
    assert result == exec_req


# ── Gate behaviour: checker engaged after first syntax failure ────────────────

async def test_checker_engaged_only_after_first_syntax_failure(agent):
    good = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": 300})

    async def fake_invoke(messages):
        mock = MagicMock()
        mock.content = good
        return mock

    agent._invoke_llm_async = fake_invoke
    review_mock = AsyncMock(
        return_value=CheckVerdict(approved=True, feedback=""))
    agent._checker_agent.review = review_mock

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    result = await agent.handle_request_iteration(_make_message(exec_payload), updater)

    assert result == good
    # First response was already valid → checker must NOT have been called.
    review_mock.assert_not_called()


async def test_checker_rejection_blocks_then_approval_sends(agent):
    good = json.dumps(
        {"kind": "exec_request", "command": "echo a", "timeout": 300})
    bad = "not json"  # first response fails syntax → engages the checker

    call_count = 0

    async def fake_invoke(messages):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        mock.content = bad if call_count == 1 else good
        return mock

    agent._invoke_llm_async = fake_invoke

    # 1st review: diagnose the syntax error. 2nd: reject a syntactically-valid
    # response. 3rd: finally approve.
    agent._checker_agent.review = AsyncMock(side_effect=[
        CheckVerdict(approved=False, feedback="diagnose"),
        CheckVerdict(approved=False, feedback="still wrong"),
        CheckVerdict(approved=True, feedback=""),
    ])

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    result = await agent.handle_request_iteration(_make_message(exec_payload), updater)
    assert result == good
    assert call_count == 3


# ── #2 exec_result output truncation ───────────────────────────────────────────

def test_truncate_field_keeps_head_and_tail(agent):
    big = "A" * 1000 + "B" * 20000 + "Z" * 1000
    out = agent._truncate_field(big, budget=6000)
    assert len(out) < len(big)
    assert out.startswith("A")
    assert out.endswith("Z")
    assert "truncated" in out


def test_truncate_exec_result_bounds_stdout(agent):
    payload = json.dumps(
        {"kind": "exec_result", "stdout": "x" * 50000, "exit_code": 0})
    out = agent._truncate_exec_result(payload)
    data = json.loads(out)
    assert len(data["stdout"]) <= MAX_OUTPUT_CHARS + 100  # + elision marker
    assert data["exit_code"] == 0


def test_truncate_exec_result_passes_small_output(agent):
    payload = json.dumps(
        {"kind": "exec_result", "stdout": "all good", "exit_code": 0})
    out = agent._truncate_exec_result(payload)
    assert json.loads(out)["stdout"] == "all good"


# ── #5 deterministic recon turn 0 ──────────────────────────────────────────────

async def test_task_triggers_recon_without_llm(agent):
    invoke_mock = AsyncMock()
    agent._invoke_llm_async = invoke_mock

    task_payload = json.dumps({"kind": "task", "instruction": "do something"})
    updater = MagicMock()
    updater.start_work = AsyncMock()

    result = await agent.handle_request_iteration(_make_message(task_payload), updater)

    parsed = json.loads(result)
    assert parsed["kind"] == "exec_request"
    assert parsed["command"] == RECON_CMD
    invoke_mock.assert_not_called()


# ── #4 run() out-of-budget sends a single final ────────────────────────────────

async def test_run_sends_final_when_out_of_turns(agent):
    agent._turn_count = agent._max_turn_count  # exhausted

    updater = MagicMock()
    updater.complete = AsyncMock()
    updater.new_agent_message = MagicMock(return_value="final-msg")

    task_payload = json.dumps({"kind": "task", "instruction": "x"})
    await agent.run(_make_message(task_payload), updater)

    updater.complete.assert_awaited_once()
    # The completed message was built from a final payload.
    sent_text = updater.new_agent_message.call_args.kwargs["parts"][0].root.text
    assert json.loads(sent_text)["kind"] == "final"
