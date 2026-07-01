import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.terminal_bench import TerminalBenchAgent
from agents.configuration.config import MAX_OUTPUT_CHARS
from agents.terminal_bench_supplementary import utils
from agents.critic import CriticAgent, CriticVerdict
from agents.planner import PlannerOutput
from agents.tools.agent_memory import AgentMemory
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


# ── CriticAgent verdict parsing (no LLM) ───────────────────────────────────────

def test_parse_verdict_approved():
    v = CriticAgent._parse_verdict('{"approved": true, "feedback": ""}')
    assert v.approved is True


def test_parse_verdict_rejected_with_feedback():
    raw = '{"approved": false, "feedback": "Send only the first command."}'
    v = CriticAgent._parse_verdict(raw)
    assert v.approved is False
    assert "first command" in v.feedback


def test_parse_verdict_embedded_in_prose():
    raw = 'Here is my verdict:\n{"approved": false, "feedback": "fix it"}\nthanks'
    v = CriticAgent._parse_verdict(raw)
    assert v.approved is False
    assert v.feedback == "fix it"


def test_parse_verdict_unparseable_does_not_approve():
    v = CriticAgent._parse_verdict("totally not json")
    assert v.approved is False


async def test_critic_falls_open_on_llm_error():
    """When the underlying LLM is unavailable, invoke() returns error=True."""
    client = MagicMock()
    client.invoke_async = AsyncMock(side_effect=RuntimeError("no api key"))
    critic = CriticAgent(client)

    v = await critic.invoke([SystemMessage(content="critic")], '{"kind":"final"}')
    assert v.error is True
    assert v.approved is False


# ── AgentMemory: plan slot + rolling window ────────────────────────────────────

def test_memory_set_plan_from_list():
    sys = SystemMessage(content="sys")
    mem = AgentMemory(sys, sys, sys, short_term_window=10)
    mem.set_plan(["a", "b", "c"])
    plan = mem.get_plan()
    assert plan is not None
    assert "a" in plan.content and "c" in plan.content


def test_memory_window_keeps_last_10_short_term():
    sys = SystemMessage(content="sys")
    mem = AgentMemory(sys, sys, sys, short_term_window=10)
    mem.set_plan(["do x"])
    # 8 exchange pairs = 16 messages; only the last 10 should survive.
    for i in range(8):
        mem.add(HumanMessage(content=f"exec_result {i}"))
        mem.add(AIMessage(content=f"response {i}"))

    messages = mem.build_planner_messages()
    # planner_system + plan + short-term header + 10 short-term messages
    assert messages[0].content == "sys"
    assert "do x" in messages[1].content
    short_term = messages[3:]
    assert len(short_term) == 10
    # Oldest surviving is exchange #3 (pairs 0,1,2 rolled off).
    assert short_term[0].content == "exec_result 3"
    assert short_term[-1].content == "response 7"


# ── Pipeline: planner runs, critic gates, approved exec is returned ────────────

async def test_planner_runs_and_approved_exec_returned(agent):
    exec_req = json.dumps(
        {"kind": "exec_request", "command": "ls -la", "timeout": 300})

    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(
            updated_plan=["explore", "build"], task_formulation="explore"))
    agent._actor_agent.invoke = AsyncMock(return_value=_ai(exec_req))
    agent._critic_agent.invoke = AsyncMock(
        return_value=CriticVerdict(approved=True, feedback=""))

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock())

    # The approved actor candidate is returned — not the plan.
    assert result == exec_req
    agent._planner_agent.invoke.assert_awaited_once()
    # The plan was stored in memory.
    stored = agent._memory.get_plan()
    assert stored is not None
    assert "explore" in stored.content


async def test_critic_runs_even_for_valid_first_candidate(agent):
    good = json.dumps(
        {"kind": "exec_request", "command": "echo hi", "timeout": 300})

    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=[], task_formulation="do it"))
    agent._actor_agent.invoke = AsyncMock(return_value=_ai(good))
    critic_mock = AsyncMock(
        return_value=CriticVerdict(approved=True, feedback=""))
    agent._critic_agent.invoke = critic_mock

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock())

    assert result == good
    # The critic is always the send-gate — it runs even for a valid candidate.
    critic_mock.assert_awaited_once()


async def test_critic_rejection_blocks_then_approval_sends(agent):
    good = json.dumps(
        {"kind": "exec_request", "command": "echo a", "timeout": 300})

    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=[], task_formulation="do it"))
    agent._actor_agent.invoke = AsyncMock(return_value=_ai(good))
    agent._critic_agent.invoke = AsyncMock(side_effect=[
        CriticVerdict(approved=False, feedback="diagnose"),
        CriticVerdict(approved=False, feedback="still wrong"),
        CriticVerdict(approved=True, feedback=""),
    ])

    exec_payload = json.dumps({"kind": "exec_result", "stdout": "", "exit_code": 0})
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock())

    assert result == good
    assert agent._critic_agent.invoke.await_count == 3
    assert agent._actor_agent.invoke.await_count == 3


# ── exec_result output truncation (no LLM) ─────────────────────────────────────

def test_truncate_field_keeps_head_and_tail():
    big = "A" * 1000 + "B" * 20000 + "Z" * 1000
    out = utils.truncate_field(big, budget=6000)
    assert len(out) < len(big)
    assert out.startswith("A")
    assert out.endswith("Z")
    assert "truncated" in out


def test_truncate_exec_result_bounds_stdout():
    payload = json.dumps(
        {"kind": "exec_result", "stdout": "x" * 50000, "exit_code": 0})
    out = utils.truncate_exec_result(payload)
    data = json.loads(out)
    assert len(data["stdout"]) <= MAX_OUTPUT_CHARS + 100  # + elision marker
    assert data["exit_code"] == 0


def test_truncate_exec_result_passes_small_output():
    payload = json.dumps(
        {"kind": "exec_result", "stdout": "all good", "exit_code": 0})
    out = utils.truncate_exec_result(payload)
    assert json.loads(out)["stdout"] == "all good"


# ── Turn budget: Agent.run sends a single final when out of turns ─────────────

async def test_agent_sends_final_when_out_of_turns():
    from agent import Agent

    a = Agent()
    a._turn_count = a._max_turn_count  # exhausted

    updater = MagicMock()
    updater.complete = AsyncMock()
    updater.new_agent_message = MagicMock(return_value="final-msg")

    task_payload = json.dumps({"kind": "task", "instruction": "x"})
    await a.run(_make_message(task_payload), updater)

    updater.complete.assert_awaited_once()
    # The completed message was built from a final payload.
    sent_text = updater.new_agent_message.call_args.kwargs["parts"][0].root.text
    assert json.loads(sent_text)["kind"] == "final"
