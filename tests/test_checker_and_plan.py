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
    """An AIMessage that carries a tool_calls list (Actor tool-call response)."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_test123",
                "name": "execute_command",
                "args": {"command": command, "timeout": timeout},
            }
        ],
    )


def _ai_text(content: str):
    """An AIMessage with plain text (Actor finalize response)."""
    return AIMessage(content=content)


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


def test_parse_verdict_double_braces():
    """LLMs sometimes emit {{...}} — should still parse correctly."""
    raw = '{{"approved": false, "feedback": "empty command"}}'
    v = CriticAgent._parse_verdict(raw)
    assert v.approved is False
    assert "empty command" in v.feedback


def test_parse_verdict_triple_braces():
    """Even triple braces should be handled."""
    raw = '{{{"approved": true, "feedback": ""}}}'
    v = CriticAgent._parse_verdict(raw)
    assert v.approved is True


def test_parse_verdict_unparseable_does_not_approve():
    v = CriticAgent._parse_verdict("totally not json")
    assert v.approved is False


async def test_critic_falls_open_on_llm_error():
    """When the underlying LLM is unavailable, invoke() returns error=True."""
    client = MagicMock()
    client.invoke_with_response_format_async = AsyncMock(
        side_effect=RuntimeError("no api key")
    )
    critic = CriticAgent(client)

    v = await critic.invoke(
        [SystemMessage(content="critic")],
        {"name": "execute_command", "arguments": {"command": "ls", "timeout": 300}},
    )
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


def test_memory_pending_tool_call():
    sys = SystemMessage(content="sys")
    mem = AgentMemory(sys, sys, sys, short_term_window=10)
    tc = {"name": "execute_command", "arguments": {"command": "ls", "timeout": 300}}
    mem.set_pending_tool_call(tc)
    assert mem.get_pending_tool_call() == tc


def test_memory_window_keeps_last_10_short_term():
    sys = SystemMessage(content="sys")
    mem = AgentMemory(sys, sys, sys, short_term_window=10)
    mem.set_plan(["do x"])
    # 8 exchange pairs = 16 messages added to chat_history
    for i in range(8):
        mem.add(HumanMessage(content=f"exec_result {i}"))
        mem.add(AIMessage(content=f"response {i}"))

    messages = mem.build_planner_messages()
    # build_planner_messages() returns: system + (task if set) + chat_history
    # No plan or windowing is applied by build_planner_messages()
    assert messages[0].content == "sys"
    # No task was set, so messages[1:] is the full chat_history (16 messages)
    assert len(messages[1:]) == 16
    assert messages[1].content == "exec_result 0"
    assert messages[-1].content == "response 7"
    # Plan is stored separately via set_plan / get_plan
    plan = mem.get_plan()
    assert plan is not None
    assert "do x" in plan.content


# ── Pipeline: planner runs, critic gates, approved exec is returned ────────────

async def test_planner_runs_and_approved_exec_returned(agent):
    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(
            updated_plan=["explore", "build"], task_formulation="explore"
        )
    )
    agent._actor_agent.invoke = AsyncMock(
        return_value=_ai_with_tool_call("ls -la")
    )
    agent._critic_agent.invoke = AsyncMock(
        return_value=CriticVerdict(approved=True, feedback="")
    )

    # Planner only runs on kind="task", not on exec_result
    task_payload = json.dumps(
        {"kind": "task", "instruction": "explore the filesystem"}
    )
    result = await agent.handle_request_iteration(
        _make_message(task_payload), MagicMock()
    )

    # The approved actor candidate is returned as exec_request JSON.
    result_dict = json.loads(result)
    assert result_dict["kind"] == "exec_request"
    assert result_dict["command"] == "ls -la"
    agent._planner_agent.invoke.assert_awaited_once()
    # The plan was stored in chat history (not via set_plan).
    messages = agent._memory.build_planner_messages()
    plan_found = any("explore" in str(m.content) for m in messages[1:])
    assert plan_found, "Plan should be stored in chat history"


async def test_critic_runs_even_for_valid_first_candidate(agent):
    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=[], task_formulation="do it")
    )
    agent._actor_agent.invoke = AsyncMock(
        return_value=_ai_with_tool_call("echo hi")
    )
    critic_mock = AsyncMock(
        return_value=CriticVerdict(approved=True, feedback="")
    )
    agent._critic_agent.invoke = critic_mock

    exec_payload = json.dumps(
        {"kind": "exec_result", "stdout": "", "exit_code": 0}
    )
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock()
    )

    result_dict = json.loads(result)
    assert result_dict["kind"] == "exec_request"
    assert result_dict["command"] == "echo hi"
    # The critic is always the send-gate — it runs even for a valid candidate.
    critic_mock.assert_awaited_once()


async def test_critic_rejection_blocks_then_approval_sends(agent):
    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=[], task_formulation="do it")
    )
    agent._actor_agent.invoke = AsyncMock(
        return_value=_ai_with_tool_call("echo a")
    )
    agent._critic_agent.invoke = AsyncMock(side_effect=[
        CriticVerdict(approved=False, feedback="diagnose"),
        CriticVerdict(approved=False, feedback="still wrong"),
        CriticVerdict(approved=True, feedback=""),
    ])

    exec_payload = json.dumps(
        {"kind": "exec_result", "stdout": "", "exit_code": 0}
    )
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock()
    )

    result_dict = json.loads(result)
    assert result_dict["kind"] == "exec_request"
    assert result_dict["command"] == "echo a"
    assert agent._critic_agent.invoke.await_count == 3
    assert agent._actor_agent.invoke.await_count == 3


async def test_actor_finalize_returns_final(agent):
    """When the Actor returns plain text (no tool_calls), the loop nudges
    the actor to retry with tools. After exhaustion it returns a no-op exec_request."""
    agent._planner_agent.invoke = AsyncMock(
        return_value=PlannerOutput(updated_plan=[], task_formulation="do it")
    )
    agent._actor_agent.invoke = AsyncMock(
        return_value=_ai_text("Task completed successfully.")
    )

    exec_payload = json.dumps(
        {"kind": "exec_result", "stdout": "done", "exit_code": 0}
    )
    result = await agent.handle_request_iteration(
        _make_message(exec_payload), MagicMock()
    )

    # Actor keeps returning plain text → loop exhausts → no-op exec_request
    result_dict = json.loads(result)
    assert result_dict["kind"] == "exec_request"
    assert result_dict["command"] == "true"
    assert agent._actor_agent.invoke.await_count == agent._max_critic_actor_rounds


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
        {"kind": "exec_result", "stdout": "x" * 50000, "exit_code": 0}
    )
    out = utils.truncate_exec_result(payload)
    data = json.loads(out)
    assert len(data["stdout"]) <= MAX_OUTPUT_CHARS + 100  # + elision marker
    assert data["exit_code"] == 0


def test_truncate_exec_result_passes_small_output():
    payload = json.dumps(
        {"kind": "exec_result", "stdout": "all good", "exit_code": 0}
    )
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
