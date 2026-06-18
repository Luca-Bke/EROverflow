"""Terminal Bench Shell v1 purple agent.

Implements the terminal-bench-shell-v1 protocol: receives a task instruction,
issues shell commands one at a time via exec_request, and signals completion
with final. State persists across turns via the shared Agent instance per context_id.
"""

import json
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable, tracing_context

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from openai import RateLimitError

from agents.llms.llm_client import TerminalBenchLLMClientInterface
from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception
from agents.tools.AgentInnerMessage import AgentInnerMessage
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.agent_memory import AgentMemory
from agents.llms.academic_cloud import AcademicCloudLLMClient
from agents.llms.open_router import OpenRouterLLMClient
from agents.tools.response_format_checker import ResponseFormatChecker
from agents.checker_agent import CheckerAgent

SYSTEM_PROMPT = """\
You are a terminal agent solving complex command-line tasks in a live shell environment.

You will receive messages as JSON. Respond ONLY with a SINGLE valid JSON object — one of:
  {"kind": "exec_request", "command": "<shell command>", "timeout": 300}
or to organise your work into a plan you will work through step by step:
  {"kind": "plan", "steps": ["step 1", "step 2", "..."]}
or when the task is complete:
  {"kind": "final"}

CRITICAL — exactly ONE JSON object per response:
- Respond with EXACTLY ONE JSON object. NEVER emit several JSON objects in one response.
- NEVER send a sequence of commands at once. Send only the FIRST command, then WAIT for its
  execution result before issuing the next one.
- Do not include any text outside the JSON object.
- Always end the task with {"kind": "final"} once it is complete — do not leave it hanging.

Planning:
- You may send {"kind": "plan", "steps": [...]} to record or update a plan. A plan is internal:
  it is NOT executed in the shell, it is stored and shown back to you so you can work through it
  step by step. After sending a plan, issue the first exec_request of that plan.

Format checking:
- A checker validates your response format. If your response is malformed (e.g. multiple commands
  in one response), it returns concrete feedback. Read it and reply with a single, corrected JSON object.

Workflow:
- Turn 0 is an automatic reconnaissance command (pwd/ls/find/git/tools). Read its output.
- Then send a {"kind": "plan", ...} grounded in what recon revealed, and work through it step by step.

Command Execution Rules:
- Never use interactive commands (vim, nano, less, ssh -t, top, htop, etc.)
- Always use non-interactive flags: apt-get -y, git --no-pager, python -c, etc.
- Bound the output of long/noisy commands so they do not flood your context:
    apt-get install -y X > /tmp/log 2>&1; tail -n 40 /tmp/log
    pip install --break-system-packages X 2>&1 | tail -3
- VERIFY before you finish: actually run the task's test/verification (e.g. the test harness,
  the smoke test, or a grep check) and confirm it passes BEFORE sending {"kind": "final"}.
- If a command fails, diagnose and try a different approach
- You can send a maximum of 30 commands total
- If the stderr and stdout of a command are not relevant to your further succeeding (e.g. the output of apt-get install update),
then pipe the output to null, the output does not clog up the history
- When possible, use filters to find the relevant information in log files or similar data

"""

# Head+tail budget (chars) for stdout/stderr of a single exec_result kept in memory.
MAX_OUTPUT_CHARS = 6000

# Fixed turn-0 reconnaissance: grounds every later decision in the real
# environment. Sent deterministically (no LLM call) on the first task message.
RECON_CMD = (
    "echo '=== PWD ===' && pwd && "
    "echo '=== LS ===' && ls -la && "
    "echo '=== FILES ===' && find . -maxdepth 2 -not -path '*/.*' -type f | sort | head -40 && "
    "echo '=== GIT ===' && (git log --oneline -5 2>/dev/null || echo '(no git)') && "
    "echo '=== TOOLS ===' && (which python3 pip git curl make 2>/dev/null | head -10 || true)"
)

LLM_PROVIDER_DICTIONARY: dict[str, TerminalBenchLLMClientInterface] = {
    "openrouter": OpenRouterLLMClient,
    "academiccloud": AcademicCloudLLMClient
}

LLM_PROVIDER = "openrouter"


class TerminalBenchAgent:
    """Purple agent for Terminal Bench 2.0.

    Maintains per-session conversation history across A2A turns.
    """

    def __init__(self) -> None:
        self._trace_enabled = bool(os.getenv("LANGSMITH_API_KEY"))
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "EROverflow-terminal-bench"
        if not os.getenv("LANGSMITH_ENDPOINT"):
            os.environ["LANGSMITH_ENDPOINT"] = "api.smith.langchain.com"

        # this is some MingXuan level code
        llm_client_class = LLM_PROVIDER_DICTIONARY.get(LLM_PROVIDER)
        self._llm_client: TerminalBenchLLMClientInterface = llm_client_class()
        print("model:", self._llm_client.model())
        self._memory = AgentMemory(SystemMessage(str(
            SYSTEM_PROMPT)), short_term_window=10)
        self._checker_agent = CheckerAgent()
        self._turn_count = 0
        self._max_turn_count = 30
        self._max_syntax_retries = 5
        self._max_plan_turns = 3

    @traceable(name="agent_session", run_type="chain")
    def _emit_session_trace(
        self,
        history: list[dict],
        turn_count: int,
        rate_limited: bool,
        retry_log: list[dict],
    ) -> dict:
        return {
            "turn_count": turn_count,
            "rate_limited": rate_limited,
            "retry_count": len(retry_log),
            "history_length": len(history),
            "completed": not rate_limited,
        }

    @staticmethod
    def _truncate_field(value: str, budget: int = MAX_OUTPUT_CHARS) -> str:
        """Keep the head and tail of a long string, eliding the middle."""
        if len(value) <= budget:
            return value
        half = budget // 2
        elided = len(value) - 2 * half
        return f"{value[:half]}\n…[{elided} chars truncated]…\n{value[-half:]}"

    def _truncate_exec_result(self, input_text: str) -> str:
        """Bound stdout/stderr of an exec_result before it enters memory.

        A single apt-get/pip/training command can emit tens of thousands of
        lines; storing that verbatim blows the rolling context window and can
        crash the A2A gateway. We keep head+tail of each stream.
        """
        try:
            data = json.loads(input_text)
        except (json.JSONDecodeError, ValueError):
            return self._truncate_field(input_text)

        for field in ("stdout", "stderr", "output"):
            if isinstance(data.get(field), str):
                data[field] = self._truncate_field(data[field])
        return json.dumps(data)

    async def handle_request_iteration(self, message: Message,
                                       updater: TaskUpdater) -> str:
        

        input_text = get_message_text(message)

        print(f"Received Message: {input_text}")

        await updater.start_work(
            new_agent_text_message(f"Turn {self._turn_count}: thinking...")
        )

        input_dict = json.loads(input_text)

        print(f"received message type: {input_dict.get("kind")}")

        if input_dict.get("kind") == "task":
            print("Received initial task instruction:",
                  input_dict.get("instruction"))
            self._memory.set_task(HumanMessage(content=input_text))
            # Turn 0: deterministic recon — no LLM call. Grounds the agent in the
            # real environment before it plans or acts.
            recon = json.dumps(
                {"kind": "exec_request", "command": RECON_CMD, "timeout": 60})
            self._memory.add(AIMessage(content=recon))
            return recon

        elif input_dict.get("kind") == "exec_result":
            print("Received execution result, updating history for next turn.")
            self._memory.add(HumanMessage(content=self._truncate_exec_result(input_text)))

        else:
            print(f"Received unknown message type: {input_dict.get('kind')}")

        messages = self._memory.build_messages()

        print("history for llm consumption: ", messages)

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
            print(f"LLM response (syntax_attempt {syntax_attempts + 1}): {response_text}")

            # ── Static syntax checker — always the first, cheap gate ──────────
            try:
                response_dict = self.validate_response(response_text)
            except terminal_bench_format_exception as e:
                last_error = e
                syntax_attempts += 1
                checker_engaged = True  # engage the checker agent from now on
                print(f"Syntax error (attempt {syntax_attempts}): {e.message}")
                verdict = await self._checker_agent.review(response_text, e.message)
                messages.append(AIMessage(content=response_text))
                messages.append(AgentInnerMessage(content=json.dumps({
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
                messages = self._memory.build_messages()
                if plan_turns >= self._max_plan_turns:
                    # Past the budget keep storing, but charge the syntax budget so a
                    # model that only ever plans cannot loop forever.
                    syntax_attempts += 1
                    messages.append(AgentInnerMessage(content=(
                        "You have planned enough. Now issue a single exec_request "
                        "for the first step of your plan.")))
                else:
                    messages.append(AgentInnerMessage(content=(
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
                    print(f"Checker rejected (attempt {syntax_attempts}): {verdict.feedback}")
                    messages.append(AIMessage(content=response_text))
                    messages.append(AgentInnerMessage(content=json.dumps({
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

    @staticmethod
    def _is_final(response_result: str) -> bool:
        try:
            return json.loads(response_result).get("kind") == "final"
        except (json.JSONDecodeError, AttributeError, ValueError):
            return False

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        # Out of turn budget: send a clean final so the executor never has to
        # respond with an empty message (which it treats as an error).
        if self._turn_count >= self._max_turn_count:
            print("Max turn count reached; sending final.")
            final_msg = json.dumps({"kind": "final"})
            await updater.complete(updater.new_agent_message(
                parts=[Part(root=TextPart(text=final_msg))]))
            return

        print("Run was called with the following message")

        response_result = await self.handle_request_iteration(message, updater)

        is_final = self._is_final(response_result)

        if is_final or self._llm_client.rate_limited():
            history = [
                {"role": getattr(m, "type", "unknown"), "content": str(getattr(m, "content", m))}
                for m in self._memory.build_messages()
            ]
            with tracing_context(enabled=self._trace_enabled):
                self._emit_session_trace(
                    history=history,
                    turn_count=self._turn_count,
                    rate_limited=self._llm_client.rate_limited(),
                    retry_log=self._llm_client.retry_log(),
                )

        if is_final:
            print("Agent signaled task completion.")

        # Send the agent response back to the A2A server. A task lives for only
        # one turn, so we must complete it exactly once — otherwise the executor
        # responds with an empty message and errors.
        response_msg = updater.new_agent_message(
            parts=[Part(root=TextPart(text=response_result))]
        )
        print(f"Submitting response for turn {self._turn_count}: {response_msg}")
        await updater.complete(response_msg)

        self._turn_count += 1


__all__ = ["TerminalBenchAgent"]
