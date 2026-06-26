"""Debug tests for the planner-actor-critic multi-agent setup.

Run levels:
  unit   – fully offline, no LLM calls
  mocked – uses AsyncMock LLM clients; tests agent wiring
  live   – calls a real LLM (requires API key in .env)
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from a2a.types import Message, Part, TextPart

from agents.actor import ActorAgent
from agents.critic import CriticAgent, CriticVerdict
from agents.planner import PlannerAgent, PlannerOutput
from agents.tools.agent_memory import AgentMemory
from agents.configuration.config import PLANNER_SYSTEM_PROMPT


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_message(text: str) -> Message:
    return Message(
        kind="message",
        role="user",
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id="debug-test-id",
    )


def _make_memory() -> AgentMemory:
    sys = SystemMessage(content="system")
    return AgentMemory(sys, sys, sys, short_term_window=5)


def _mock_llm(response_text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    client.invoke_async = AsyncMock(return_value=msg)
    client.rate_limited = MagicMock(return_value=False)
    client.retry_log = MagicMock(return_value=[])
    return client


# ── Unit: _split_agent_response (no LLM) ─────────────────────────────────────

def test_split_clean_json():
    raw = json.dumps({
        "updated_plan": ["explore filesystem", "read flag file"],
        "task_formulation": "List the contents of / to find the flag directory.",
    })
    msg = MagicMock()
    msg.content = raw
    out = PlannerAgent._split_agent_response(msg)
    assert out.updated_plan == ["explore filesystem", "read flag file"]
    assert "List the contents" in out.task_formulation


def test_split_json_in_markdown_code_fence():
    raw = (
        "```json\n"
        '{"updated_plan": ["step 1"], "task_formulation": "Do step 1."}\n'
        "```"
    )
    msg = MagicMock()
    msg.content = raw
    out = PlannerAgent._split_agent_response(msg)
    assert out.updated_plan == ["step 1"]
    assert out.task_formulation == "Do step 1."


def test_split_json_embedded_in_prose():
    raw = (
        'Here is my analysis.\n'
        '{"updated_plan": ["a", "b"], "task_formulation": "Run ls."}\n'
        'Hope that helps.'
    )
    msg = MagicMock()
    msg.content = raw
    out = PlannerAgent._split_agent_response(msg)
    assert out.updated_plan == ["a", "b"]
    assert "ls" in out.task_formulation


def test_split_non_list_plan_is_wrapped():
    raw = json.dumps({
        "updated_plan": "just a string plan",
        "task_formulation": "Do something.",
    })
    msg = MagicMock()
    msg.content = raw
    out = PlannerAgent._split_agent_response(msg)
    assert isinstance(out.updated_plan, list)
    assert out.updated_plan[0] == "just a string plan"


def test_split_malformed_response_falls_back():
    msg = MagicMock()
    msg.content = "I could not decide what to do next."
    out = PlannerAgent._split_agent_response(msg)
    assert isinstance(out, PlannerOutput)
    assert out.updated_plan == []
    assert "could not decide" in out.task_formulation


def test_split_missing_fields_returns_empty_defaults():
    msg = MagicMock()
    msg.content = '{"other_key": "value"}'
    out = PlannerAgent._split_agent_response(msg)
    assert out.updated_plan == []
    assert out.task_formulation == ""


# ── Mocked: planner invocation wiring ────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_task_message_stores_formulation_and_returns_output():
    planner_response = json.dumps({
        "updated_plan": ["recon", "exploit"],
        "task_formulation": "Run ls -la to list current directory.",
    })
    memory = _make_memory()
    agent = PlannerAgent(_mock_llm(planner_response))

    task = json.dumps({"kind": "task", "instruction": "Find the flag."})
    out = await agent.invoke(_make_message(task), memory)

    assert out.updated_plan == ["recon", "exploit"]
    assert "ls -la" in out.task_formulation
    # task_formulation is stored in memory before the LLM is called
    assert memory.get_subtask_formulation() is not None


@pytest.mark.asyncio
async def test_planner_exec_result_goes_into_short_term_memory():
    planner_response = json.dumps({
        "updated_plan": ["done"],
        "task_formulation": "Read the flag.",
    })
    memory = _make_memory()
    agent = PlannerAgent(_mock_llm(planner_response))

    exec_result = json.dumps({
        "kind": "exec_result",
        "stdout": "flag{test}",
        "exit_code": 0,
    })
    await agent.invoke(_make_message(exec_result), memory)

    # The exec result should appear in the short-term window
    planner_msgs = memory.build_planner_messages()
    short_term_contents = [getattr(m, "content", "") for m in planner_msgs]
    assert any("flag{test}" in c for c in short_term_contents)


@pytest.mark.asyncio
async def test_planner_result_feeds_actor_messages():
    """Plan and task_formulation from the planner must appear in actor messages."""
    planner_response = json.dumps({
        "updated_plan": ["step A", "step B"],
        "task_formulation": "Check /etc/passwd for clues.",
    })
    memory = AgentMemory(
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        SystemMessage(content="actor sys"),
        SystemMessage(content="critic sys"),
        short_term_window=5,
    )
    planner = PlannerAgent(_mock_llm(planner_response))

    task = json.dumps({"kind": "task", "instruction": "Do the thing."})
    out = await planner.invoke(_make_message(task), memory)

    memory.set_plan(out.updated_plan)
    memory.set_subtask_formulation(out.task_formulation)

    actor_msgs = memory.build_actor_messages()
    all_content = " ".join(getattr(m, "content", "") for m in actor_msgs)
    assert "step A" in all_content
    assert "/etc/passwd" in all_content


@pytest.mark.asyncio
async def test_critic_approves_valid_exec_request():
    exec_request = json.dumps({
        "kind": "exec_request",
        "command": "ls -la /",
        "timeout": 30,
    })
    critic_response = json.dumps({"approved": True, "feedback": ""})
    actor_msg = MagicMock()
    actor_msg.content = exec_request

    memory = _make_memory()
    memory.set_execution_request_candidate(actor_msg)

    critic = CriticAgent(_mock_llm(critic_response))
    verdict = critic.invoke(
        memory.build_critic_messages(),
        memory.get_execution_request_candidate(),
    )
    assert verdict.approved is True


# ── Live: calls real LLM (skip if no API key) ────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY")
    and not os.getenv("ACADEMICCLOUD_API_KEY")
    and not os.getenv("LLMHUB_APIKEY"),
    reason="No LLM API key found — set one of OPENROUTER_API_KEY / ACADEMICCLOUD_API_KEY / LLMHUB_APIKEY",
)
@pytest.mark.asyncio
async def test_live_planner_returns_valid_output():
    """Calls the configured LLM provider once and prints the planner output."""
    from agents.agent import _build_llm_client  # noqa: PLC0415

    client = _build_llm_client()
    memory = AgentMemory(
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        SystemMessage(content="actor"),
        SystemMessage(content="critic"),
        short_term_window=5,
    )
    planner = PlannerAgent(client)

    task = json.dumps({
        "kind": "task",
        "instruction": "Find all .txt files in /tmp and print their contents.",
    })
    out = await planner.invoke(_make_message(task), memory)

    print("\n── Planner output ──")
    print("updated_plan:", out.updated_plan)
    print("task_formulation:", out.task_formulation)

    assert isinstance(out.updated_plan, list), "updated_plan must be a list"
    assert len(out.updated_plan) > 0, "updated_plan must not be empty"
    assert isinstance(out.task_formulation, str), "task_formulation must be a string"
    assert len(out.task_formulation) > 10, "task_formulation looks too short"
