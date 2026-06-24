"""Terminal Bench Shell v1 purple agent.

Implements the terminal-bench-shell-v1 protocol: receives a task instruction,
issues shell commands one at a time via exec_request, and signals completion
with final. State persists across turns via the shared Agent instance per context_id.
"""

import json
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import tracing_context

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from openai import RateLimitError

from agents.llm_clients.llm_client import TerminalBenchLLMClientInterface
from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception
from agents.terminal_bench_supplementary.pipeline_messages import ExecutionRequestCandidateMessage
from agents.terminal_bench_supplementary import utils
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.agent_memory import AgentMemory
from agents.tools.response_format_checker import ResponseFormatChecker
from agents.checker_agent import CheckerAgent


class TerminalBenchAgent:
    """Purple agent for Terminal Bench 2.0.

    Maintains per-session conversation history across A2A turns.
    All configuration is injected via the constructor; nothing is read from
    module-level globals or environment variables except the LangSmith toggle.
    """

    def __init__(
        self,
        llm_client: TerminalBenchLLMClientInterface,
        system_prompt: str,
        recon_cmd: str,
        max_turn_count: int = 30,
        max_syntax_retries: int = 5,
        max_plan_turns: int = 3,
        short_term_window: int = 10,
    ) -> None:
        self._trace_enabled = bool(os.getenv("LANGSMITH_API_KEY"))
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "EROverflow-terminal-bench"
        if not os.getenv("LANGSMITH_ENDPOINT"):
            os.environ["LANGSMITH_ENDPOINT"] = "api.smith.langchain.com"

        self._llm_client = llm_client
        self._recon_cmd = recon_cmd
        self._max_turn_count = max_turn_count
        self._max_syntax_retries = max_syntax_retries
        self._max_plan_turns = max_plan_turns

        # Temporary: single prompt passed to all three roles until the
        # planner-actor-critic flow is wired up.
        _prompt = SystemMessage(system_prompt)
        self._memory = AgentMemory(_prompt, _prompt, _prompt, short_term_window=short_term_window)
        self._checker_agent = CheckerAgent()
        self._turn_count = 0

    async def handle_request_iteration(self, message: Message,
                                       updater: TaskUpdater) -> str:

        input_text = get_message_text(message)

        await updater.start_work(
            new_agent_text_message(f"Turn {self._turn_count}: thinking...")
        )

        input_dict = json.loads(input_text)

        if input_dict.get("kind") == "task":
            self._memory.set_task_formulation(HumanMessage(content=input_text))
            # Turn 0: deterministic recon — no LLM call. Grounds the agent in the
            # real environment before it plans or acts.
            recon = json.dumps(
                {"kind": "exec_request", "command": self._recon_cmd, "timeout": 60})
            self._memory.add(AIMessage(content=recon))
            return recon

        elif input_dict.get("kind") == "exec_result":
            self._memory.add(HumanMessage(
                content=utils.truncate_exec_result(input_text)))

        else:
            print(f"Received unknown message type: {input_dict.get('kind')}")

        messages = self._memory.build_actor_messages()

        last_error: terminal_bench_format_exception | None = None
        syntax_attempts = 0
        plan_turns = 0
        checker_engaged = False

        while syntax_attempts < self._max_syntax_retries:
            try:
                result = await self._llm_client.invoke_async(messages)
            except RateLimitError:
                if self._llm_client.rate_limited():
                    print("Rate limit was previously hit; returning final.")
                    return json.dumps({"kind": "final"})
                raise

            response_text = getattr(result, "content", str(result))

            # ── Static syntax checker — always the first, cheap gate ──────────
            try:
                response_dict = self.validate_response(response_text)
            except terminal_bench_format_exception as e:
                last_error = e
                syntax_attempts += 1
                checker_engaged = True  # engage the checker agent from now on
                verdict = await self._checker_agent.review(response_text, e.message)
                messages.append(AIMessage(content=response_text))
                messages.append(ExecutionRequestCandidateMessage(content=json.dumps({
                    "kind": "error",
                    "error": verdict.feedback or e.message,
                })))
                continue

            kind = response_dict.get("kind")

            # ── Internal plan turn — stored, never sent to the green agent ────
            if kind == "plan":
                self._memory.set_plan(response_dict)
                self._memory.add(AIMessage(content=response_text))
                plan_turns += 1
                messages = self._memory.build_actor_messages()
                if plan_turns >= self._max_plan_turns:
                    # Past the budget keep storing, but charge the syntax budget so a
                    # model that only ever plans cannot loop forever.
                    syntax_attempts += 1
                    messages.append(ExecutionRequestCandidateMessage(content=(
                        "You have planned enough. Now issue a single exec_request "
                        "for the first step of your plan.")))
                else:
                    messages.append(ExecutionRequestCandidateMessage(content=(
                        "Plan stored. Now issue a single exec_request for the first "
                        "step, or update the plan.")))
                continue

            # ── exec_request / final passed syntax — checker is the send gate ─
            # Cheap path: if the checker was never engaged (first response valid),
            # skip the checker entirely. Once engaged, the checker approves sends.
            # If the checker LLM is unavailable, fall open: syntax already passed.
            if checker_engaged:
                verdict = await self._checker_agent.review(response_text, None)
                if not verdict.approved and not verdict.error:
                    syntax_attempts += 1
                    messages.append(AIMessage(content=response_text))
                    messages.append(ExecutionRequestCandidateMessage(content=json.dumps({
                        "kind": "error",
                        "error": verdict.feedback,
                    })))
                    continue

            # For exec_request, send the normalised dict (clamped timeout) rather
            # than the raw model text. final passes through unchanged.
            outgoing = json.dumps(response_dict) if kind == "exec_request" else response_text
            self._memory.add(AIMessage(content=outgoing))
            return outgoing

        raise last_error

    def validate_response(self, response_text: str) -> dict:
        """Static syntax validation. Returns the parsed dict or raises.

        Recognises three kinds: exec_request, final, plan. exec_request is
        additionally checked for shell-syntax/interactive/destructive issues.
        """
        response_dict = ResponseFormatChecker.check_agent_response_valid_json(
            response_text)

        kind = response_dict.get("kind")
        if kind == "exec_request":
            ExecRequestChecker.check_exec_request(response_dict)
        elif kind == "final":
            pass
        elif kind == "plan":
            steps = response_dict.get("steps") or response_dict.get("plan")
            if not steps:
                raise terminal_bench_format_exception(
                    "plan response must include a non-empty 'steps' list: " + response_text)
        else:
            raise terminal_bench_format_exception(
                "LLM response JSON missing 'kind' field or has unknown kind: " + response_text)

        return response_dict

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        # Out of turn budget: send a clean final so the executor never has to
        # respond with an empty message (which it treats as an error).
        if self._turn_count >= self._max_turn_count:
            print("Max turn count reached; sending final.")
            final_msg = json.dumps({"kind": "final"})
            await updater.complete(updater.new_agent_message(
                parts=[Part(root=TextPart(text=final_msg))]))
            return

        response_result = await self.handle_request_iteration(message, updater)

        if utils.is_final_response(response_result) or self._llm_client.rate_limited():
            history = [
                {"role": getattr(m, "type", "unknown"),
                 "content": str(getattr(m, "content", m))}
                for m in self._memory.build_actor_messages()
            ]
            with tracing_context(enabled=self._trace_enabled):
                utils.emit_session_trace(
                    history=history,
                    turn_count=self._turn_count,
                    rate_limited=self._llm_client.rate_limited(),
                    retry_log=self._llm_client.retry_log(),
                )

        # Send the agent response back to the A2A server. A task lives for only
        # one turn, so we must complete it exactly once — otherwise the executor
        # responds with an empty message and errors.
        response_msg = updater.new_agent_message(
            parts=[Part(root=TextPart(text=response_result))]
        )
        await updater.complete(response_msg)

        self._turn_count += 1


__all__ = ["TerminalBenchAgent"]
