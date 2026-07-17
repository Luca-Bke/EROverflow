"""Terminal Bench Shell v1 purple agent — tool-call edition.

Implements the terminal-bench-shell-v1 protocol: receives a task instruction,
issues shell commands one at a time via the ``execute_command`` tool call,
and signals completion with a plain text finalize response.

The Actor uses native LLM tool calls instead of JSON strings. The Critic
runs as middleware: it intercepts every tool call, runs static + LLM safety
checks, and either approves (command is executed) or rejects (feedback is
returned as a tool response for the Actor to retry).
"""

import json
import traceback
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from langsmith import traceable
from openai import RateLimitError
from openai import APITimeoutError

from agents.actor import ActorAgent
from agents.critic import CriticAgent, CriticVerdict
from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.planner import PlannerAgent
from agents.terminal_bench_supplementary import utils
from agents.terminal_bench_supplementary.pipeline_messages import (
    ToolResponseMessage,
)
from agents.tools.agent_memory import AgentMemory
from agents.tools.tool_definitions import ACTOR_TOOLS
from a2a.utils import get_message_text, new_agent_text_message


class TerminalBenchAgent:
    """Purple agent for Terminal Bench 2.0 with tool-call architecture.

    Maintains per-session conversation history across A2A turns.
    All configuration is injected via the constructor; nothing is read from
    module-level globals or environment variables except the LangSmith toggle.
    """

    def __init__(
        self,
        llm_client: AbstractLLMClient,
        planner_system_prompt: str,
        critic_system_prompt: str,
        actor_system_prompt: str,
        max_critic_actor_rounds: int = 10,
        short_term_window: int = 10,
    ) -> None:

        self._llm_client = llm_client
        self._max_critic_actor_rounds = max_critic_actor_rounds
        self._updater: TaskUpdater | None = None

        self._memory = AgentMemory(
            planner_system_prompt=planner_system_prompt,
            actor_system_prompt=actor_system_prompt,
            critic_system_prompt=critic_system_prompt,
            short_term_window=short_term_window,
        )
        self._critic_agent = CriticAgent(llm_client)
        self._planner_agent = PlannerAgent(llm_client)
        self._actor_agent = ActorAgent(llm_client)
        self._turn_count = 0

    # ── Actor-Critic Loop ────────────────────────────────────────────────────

    @traceable(name="Actor Critic Loop", run_type="chain")
    async def __run_actor_critic_loop__(self) -> str | None:
        """Run the Actor-Critic loop using tool calls.

        Returns:
            The exec_request JSON string if the Critic approves a command.
            None if the Actor signals finalize or the loop is exhausted.
        """
        for _ in range(self._max_critic_actor_rounds):
            # ── 1. Actor invokes with tools ──────────────────────────────
            actor_messages = self._memory.build_actor_messages()
            # print(f"actor messages:\n{actor_messages}\n")

            actor_result = await self._actor_agent.invoke(
                actor_messages, ACTOR_TOOLS
            )
            await self._send_heartbeat("actor done")
            # print(f"actor result:\n{actor_result}\n")

            # ── 2. Check: tool call or plain text (finalize)? ────────────
            tool_calls = getattr(actor_result, "tool_calls", None) or []

            if not tool_calls:
                content = getattr(actor_result, "content", "")
                print(f"Actor returned response without tool calls — treating as final answer: {content!r}")
                self._memory.add(actor_result)
                return None

            # Take only the first tool call (ignore extras)
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name", "")
            tool_call_id = tool_call.get("id", str(uuid.uuid4()))
            # LangChain uses 'args'; raw OpenAI uses 'arguments' — handle both
            arguments: dict = tool_call.get("args") or tool_call.get("arguments", {})

            # ── 2b. submit_final → task complete ─────────────────────────
            if tool_name == "submit_final":
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                output = arguments.get("output", "")
                print(f"Actor submitted final result: {output!r}")
                self._memory.add(actor_result)
                return None

            # ── 2c. execute_command (or any other tool) → critic review ──
            # Ensure arguments is a dict (some LLMs return JSON strings)
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    print(f"Invalid tool call arguments (not JSON): {arguments!r}")
                    # Send error as tool response and retry
                    error_tool_msg = ToolResponseMessage(
                        content=(
                            "Tool call arguments are not valid JSON. "
                            "Use {\"command\": \"...\", \"timeout\": 300}."
                        ),
                        tool_call_id=tool_call_id,
                    )
                    self._memory.add_tool_call_and_response(
                        actor_result, error_tool_msg
                    )
                    continue

            self._memory.set_pending_tool_call(tool_call)

            # ── 3. Critic reviews the tool call ──────────────────────────
            critic_messages = self._memory.build_critic_messages()
            # print(f"critic messages:\n{critic_messages}\n")

            print(
                f"Tool call to be judged by the critic:\n"
                f"  command: {arguments.get('command')!r}\n"
                f"  timeout: {arguments.get('timeout')}\n"
            )

            critic_verdict: CriticVerdict = await self._critic_agent.invoke(
                critic_messages, tool_call
            )
            await self._send_heartbeat("critic done")
            # print(f"critic result:\n{critic_verdict}\n")

            # ── 4. Handle verdict ────────────────────────────────────────
            if critic_verdict.approved:
                # Build the exec_request JSON for the green agent
                command = arguments.get("command", "")
                timeout = 300  # always override timeout to 300
                exec_request = json.dumps({
                    "kind": "exec_request",
                    "command": command,
                    "timeout": timeout,
                })

                # Store tool call in memory with a placeholder tool response.
                # The actual exec_result will replace this placeholder on the
                # next turn when it arrives from the green agent.
                approved_tool_msg = ToolResponseMessage(
                    content="",
                    tool_call_id=tool_call_id,
                )
                self._memory.add_tool_call_and_response(
                    actor_result, approved_tool_msg
                )

                print(f"Approved exec request: {exec_request}")
                return exec_request
            else:
                # Critic rejected — send feedback as tool response for retry
                feedback = critic_verdict.feedback or "Command rejected."
                print(f"Critic rejected: {feedback}")

                reject_tool_msg = ToolResponseMessage(
                    content=(
                        f"Command rejected by Critic: {feedback}\n"
                        "Please call execute_command with a corrected command."
                    ),
                    tool_call_id=tool_call_id,
                )
                self._memory.add_tool_call_and_response(
                    actor_result, reject_tool_msg
                )
                # Loop continues — Actor will see the tool response and retry

        # Loop exhausted
        return None

    # ── Main Turn Handler ────────────────────────────────────────────────────

    @traceable(name="Turn", run_type="chain")
    @utils.TimeTracer.timed("TerminalBenchAgent.handle_request_iteration")
    async def handle_request_iteration(
        self, message: Message, updater: TaskUpdater
    ) -> str:
        self._turn_count += 1
        self._updater = updater

        try:
            input_text = get_message_text(message)
            input_dict = json.loads(input_text)

            if input_dict.get("kind") == "task":
                self._memory.set_task(HumanMessage(content=input_text))

                # ── Planner nur beim ersten Turn ──────────────────────
                planner_messages = self._memory.build_planner_messages()
                # print(f"planner messages:\n{planner_messages}\n")

                planner_output = await self._planner_agent.invoke(
                    planner_messages
                )
                await self._send_heartbeat("planner done")

                plan_content = json.dumps(planner_output.updated_plan, indent=2)
                self._memory.add(AIMessage(
                    content=f"[Plan for solving given Task]\n{plan_content}"
                ))

                # print(f"planner result plan:\n{planner_output.updated_plan}\n")
                # print(f"planner result task:\n{planner_output.task_formulation}\n")
                # ───────────────────────────────────────────────────────

            elif input_dict.get("kind") == "exec_result":
                # Replace the placeholder tool response with the actual
                # execution result, so the Actor sees it as a proper
                # tool response to its execute_command call.
                truncated = utils.truncate_exec_result(input_text)
                pending = self._memory.get_pending_tool_call()
                if pending:
                    tool_call_id = pending.get("id", str(uuid.uuid4()))
                    # Find and replace the last placeholder ToolMessage
                    # (empty content) in chat history with the real result.
                    for i in range(len(self._memory._chat_history) - 1, -1, -1):
                        msg = self._memory._chat_history[i]
                        if isinstance(msg, ToolResponseMessage) and not msg.content:
                            self._memory._chat_history[i] = ToolResponseMessage(
                                content=truncated,
                                tool_call_id=tool_call_id,
                            )
                            break
                    # Clear pending tool call after processing exec_result
                    self._memory.set_pending_tool_call(None)
                else:
                    # Fallback: no pending tool call, store as HumanMessage
                    self._memory.add(
                        HumanMessage(content=truncated)
                    )
            else:
                print(
                    f"Received unknown message type: {input_dict.get('kind')}"
                )

            # ── Actor-Critic Loop (in jedem Turn) ──────────────────────
            exec_request = await self.__run_actor_critic_loop__()
            if exec_request is not None:
                return exec_request

        except (RateLimitError, APITimeoutError):
            print("Rate limit or timeout hit; returning final.")
            return json.dumps({"kind": "final"})
        except Exception as e:
            error_msg = str(e).lower()
            # Treat "empty generation" errors as final answer (LLM considers task done)
            if "empty" in error_msg and ("generation" in error_msg or "output" in error_msg):
                print(f"LLM returned empty response ({e}) — treating as final answer")
                return json.dumps({"kind": "final"})
            print(
                "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
            )
            raise

        # Actor finalized or loop exhausted
        print(
            f"Actor finalized or critic did not approve any candidate "
            f"within {self._max_critic_actor_rounds} rounds; returning final."
        )
        return json.dumps({"kind": "final"})

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        return await self.handle_request_iteration(message, updater)

    async def _send_heartbeat(self, status: str) -> None:
        """Send a minimal status update to keep the SSE stream alive."""
        if self._updater is not None:
            try:
                await self._updater.start_work(
                    new_agent_text_message(f"... {status} ...")
                )
            except Exception as e:
                print(f"Heartbeat failed (non-fatal): {e}")


__all__ = ["TerminalBenchAgent"]
