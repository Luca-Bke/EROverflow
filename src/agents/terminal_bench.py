"""Terminal Bench Shell v1 purple agent.

Implements the terminal-bench-shell-v1 protocol: receives a task instruction,
issues shell commands one at a time via exec_request, and signals completion
with final. State persists across turns via the shared Agent instance per context_id.
"""

import json
import os
import asyncio
import re
import subprocess
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable, tracing_context

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception
from agents.tools.AgentInnerMessage import AgentInnerMessage
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.agent_memory import AgentMemory
from agents.tools.response_format_checker import ResponseFormatChecker

SYSTEM_PROMPT = """\
You are a terminal agent solving complex command-line tasks in a live shell environment.

You will receive messages as JSON. Respond ONLY with valid JSON — either:
  {"kind": "exec_request", "command": "<shell command>", "timeout": 30}
or when the task is complete:
  {"kind": "final"}
  
The following rules apply to the JSON request you ought to send:
- Only one execution request may be performed at a time, do not generate a response with two consecutive json requests
- Do not include any text outside the JSON object
- Do not send two commands in a row without waiting for the execution result and updating your history
- Always ensure to end with the final command that completes the task, do not leave the task hanging without signaling completion


Command Execution Rules:
- Never use interactive commands (vim, nano, less, ssh -t, top, htop, etc.)
- Always use non-interactive flags: apt-get -y, git --no-pager, python -c, etc.
- Verify your work before sending final (run the test or check the output)
- If a command fails, diagnose and try a different approach
- You can send a maximum of 30 commands total
- If the stderr and stdout of a command are not relevant to your further succeeding (e.g. the output of apt-get install update), 
then pipe the output to null, the output does not clog up the history
- When possible, use filters to find the relevant information in log files or similar data 

"""


class TerminalBenchAgent:
    """Purple agent for Terminal Bench 2.0.

    Maintains per-session conversation history across A2A turns.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = os.getenv("ACADEMICCLOUD_MODEL", "qwen3-coder-30b-a3b-instruct")
        print("model:", self._model)
        self._base_url = os.getenv("ACADEMICCLOUD_ENDPOINT",
                                   "https://chat-ai.academiccloud.de/v1")
        self._trace_enabled = bool(os.getenv("LANGSMITH_API_KEY"), False)
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "EROverflow-terminal-bench"
        if not os.getenv("LANGSMITH_ENDPOINT"):
            os.environ["LANGSMITH_ENDPOINT"] = "api.smith.langchain.com"

        self._llm: ChatOpenAI | None = None
        self._memory = AgentMemory(SystemMessage(str(
            SYSTEM_PROMPT)), short_term_window=30)
        self._turn_count = 0
        self._max_turn_count = 30
        self._max_syntax_retries = 5
        self._temperature = 0.7

        self._boundary_logged = False
        self._rate_limited = False
        self._backoff_enabled = os.getenv(
            "ENABLE_RATE_LIMIT_BACKOFF", "true").lower() in ("1", "true", "yes", "True")
        self._backoff_max_retries = int(os.getenv("BACKOFF_MAX_RETRIES", "4"))
        self._backoff_base_delay = float(
            os.getenv("BACKOFF_BASE_DELAY", "5.0"))

    @traceable(name="green_purple_boundary", run_type="chain")
    def _trace_boundary(self, input_text: str, response_text: str) -> dict[str, str]:
        if self._boundary_logged:
            return {"skipped": "true"}
        self._boundary_logged = True
        return {
            "event": "trace_boundary",
            "source": "green_agent",
            "via": "terminal_bench",
            "target": "green_agent",
            "start_payload": input_text,
            "end_payload": response_text,
        }

    def _create_llm(self) -> ChatOpenAI:
        """Create or get the ChatOpenAI instance for AcademicCloud."""
        if self._llm:
            return self._llm

        api_key = self._get_academic_cloud_api_key()

        self._llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url=self._base_url,
            temperature=self._temperature,
            tags=["eroverflow", "terminal-bench"],
            metadata={"agent": "terminal_bench", "provider": "academiccloud"},
        )
        return self._llm

    @traceable(run_type="llm")
    async def _invoke_llm_async(self, messages: list[Any]) -> str:
        """Invoke the LLM. On 429, sets _rate_limited and re-raises.
        If ENABLE_RATE_LIMIT_BACKOFF is set, retries with exponential backoff first."""
        llm = self._create_llm()
        loop = asyncio.get_running_loop()
        max_attempts = self._backoff_max_retries if self._backoff_enabled else 1

        for attempt in range(max_attempts):
            try:
                result = await loop.run_in_executor(None, lambda: llm.invoke(messages))
                return result
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in str(e)
                    or "rate limit" in err_str
                    or getattr(e, "status_code", None) == 429
                )
                if is_rate_limit and self._backoff_enabled and attempt < max_attempts - 1:
                    delay = self._backoff_base_delay * (2 ** attempt)
                    print(
                        f"Rate limit hit (attempt {attempt + 1}/{max_attempts}), retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)
                    continue
                if is_rate_limit:
                    self._rate_limited = True
                raise e

    def _get_academic_cloud_api_key(self):
        api_key = os.getenv("ACADEMICCLOUD_API_KEY")

        if not api_key:
            raise ValueError(
                "ACADEMICCLOUD_API_KEY environment variable not set")

        return api_key

    async def handle_request_iteration(self, message: Message,
                                       updater: TaskUpdater) -> str:
        if self._rate_limited:
            print("Rate limit was previously hit; returning final immediately.")
            return json.dumps({"kind": "final"})

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

        elif input_dict.get("kind") == "exec_result":
            print("Received execution result, updating history for next turn.")
            self._memory.add(HumanMessage(content=input_text))

        else:
            print(f"Received unknown message type: {input_dict.get('kind')}")

        messages = self._memory.build_messages()

        print("history for llm consumption: ", messages)

        last_error: terminal_bench_format_exception | None = None
        for attempt in range(self._max_syntax_retries):
            try:
                result = await self._invoke_llm_async(messages)
            except Exception as e:
                if self._rate_limited:
                    print("API rate limit exhausted; signaling final.")
                    return json.dumps({"error": "API Rate limit exceeded"})
                raise e

            response_text = getattr(result, "content", str(result))
            print(f"LLM response (attempt {attempt + 1}): {response_text}")

            try:
                response_text = self.postprocess_response(
                    response_text, updater)

                self._memory.add(AIMessage(content=response_text))
                return response_text

            except terminal_bench_format_exception as e:
                last_error = e
                print(f"Format error on attempt {attempt + 1}: {e.message}")
                messages.append(AIMessage(content=response_text))
                messages.append(AgentInnerMessage(content=json.dumps({
                    "kind": "error",
                    "error": e.message,
                })))

        raise last_error

    def postprocess_response(self, response_text: str, updater: TaskUpdater) -> str:
        """Post-process the LLM response to ensure it's valid JSON with expected structure."""

        # here we should have some sanity checks to ensure the response is
        # valid JSON with the expected structure and the command looks
        # reasonable (not rm -rf / or something) before
        # we send it back to the A2A server

        response_dict = ResponseFormatChecker.check_agent_response_valid_json(
            response_text)

        if response_dict.get("kind") == "exec_request":
            ExecRequestChecker.check_exec_request(response_dict)

        elif (response_dict.get("kind") == "final"):
            pass  # fine as well
        else:
            # error handling / break to rerun or fix logic
            raise terminal_bench_format_exception(
                "LLM response JSON missing 'kind' field or has unknown kind: " + response_text)

        return response_text  # valid response, pass through to A2A server

    @traceable
    async def run(self, message: Message, updater: TaskUpdater) -> None:

        if self._turn_count < self._max_turn_count:
            print("Run was called with the following message")

            response_result = await self.handle_request_iteration(message, updater)

            if (response_result == json.dumps({"kind": "final"})):
                print("Agent signaled task completion.")
                # signal completion to A2A server
                updater.complete(updater.new_agent_message(
                    [Part(root=TextPart(text=response_result))]))

            # send agent response back to A2A server
            response_msg = updater.new_agent_message(
                parts=[Part(root=TextPart(text=response_result))]
            )

            print(
                f"Submitting response for turn {self._turn_count}: {response_msg}")

            with tracing_context(enabled=self._trace_enabled):
                self._trace_boundary(
                    get_message_text(message), response_msg)

            # our task lives for only one turn, we need to complete it
            # otherwise the executor will respond with an empty message for us => error
            await updater.complete(response_msg)

            self._turn_count += 1


__all__ = ["TerminalBenchAgent"]
