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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable, tracing_context

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception

SYSTEM_PROMPT = """\
You are a terminal agent solving command-line tasks in a live shell environment.

You will receive messages as JSON. Respond ONLY with valid JSON — either:
  {"kind": "exec_request", "command": "<shell command>", "timeout": 30}
or when the task is complete:
  {"kind": "final"}

Rules:
- Never use interactive commands (vim, nano, less, ssh -t, top, htop, etc.)
- Always use non-interactive flags: apt-get -y, git --no-pager, python -c, etc.
- Verify your work before sending final (run the test or check the output)
- If a command fails, diagnose and try a different approach
- Maximum 30 commands total
- Do not include any text outside the JSON object
- Do not send two commands in a row without waiting for the execution result and updating your history
- Always ensure to end with the final command that completes the task, do not leave the task hanging without signaling completion
"""


class TerminalBenchAgent:
    """Purple agent for Terminal Bench 2.0.

    Maintains per-session conversation history across A2A turns.
    """

    # Always interactive — no non-interactive mode exists
    _ALWAYS_INTERACTIVE = frozenset({
        "vim", "vi", "nvim", "nano", "emacs", "pico",
        "less", "more", "man",
        "top", "htop", "btop",
        "ssh",
        "mysql", "psql",
    })

    # Interactive only when called without arguments (bare REPL invocation)
    _REPL_COMMANDS = frozenset({"python", "python3", "node", "irb", "iex"})

    _DESTRUCTIVE_PATTERNS = [
        r"rm\s+-[^\s]*r[^\s]*\s+/",   # rm -rf /
        r"dd\s+.*of=/dev/[sh]d",       # dd onto block device
        r":\(\)\s*\{.*\}",             # fork bomb
        r"mkfs\.",                      # format filesystem
        r">\s*/dev/[sh]d",             # write directly to block device
    ]

    def __init__(self, model: str | None = None) -> None:
        self._model = os.getenv("ACADEMICCLOUD_MODEL",
                                "meta-llama-3.1-8b-instruct")
        self._base_url = os.getenv("ACADEMICCLOUD_ENDPOINT",
                                   "https://chat-ai.academiccloud.de/v1")
        self._trace_enabled = False  # bool(os.getenv("LANGSMITH_API_KEY"))
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "EROverflow-terminal-bench"

        self._llm: ChatOpenAI | None = None
        self._history: list[Any] = []
        self._turn_count = 0
        self._max_turn_count = 10
        self._max_syntax_retries = 3
        self._temperature = 0.7

        self._boundary_logged = False

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
        """Invoke the LLM in a non-blocking way using asyncio."""
        llm = self._create_llm()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: llm.invoke(messages))
        return result

    def _get_academic_cloud_api_key(self):
        api_key = os.getenv("ACADEMICCLOUD_API_KEY")

        if not api_key:
            raise ValueError(
                "ACADEMICCLOUD_API_KEY environment variable not set")

        return api_key

    async def handle_request_iteration(self, message: Message,
                                       updater: TaskUpdater) -> str:
        input_text = get_message_text(message)

        # this is just to not waist api calls
        if (self._history is not None):
            if (len(self._history) >= 2):
                if (input_text == self._history[-2].content):
                    print("Received duplicate message, skipping processing.")
                    # or some other appropriate response or handling
                    return json.dumps({"kind": "final"})

        print(f"Received Message: {input_text}")

        await updater.start_work(
            new_agent_text_message(f"Turn {self._turn_count}: thinking...")
        )

        input_dict = json.loads(input_text)

        print(f"received message type: {input_dict.get("kind")}")

        if input_dict.get("kind") == "task":
            print("Received initial task instruction:",
                  input_dict.get("instruction"))

            self._history.append(HumanMessage(content=input_text))

        elif input_dict.get("kind") == "exec_result":
            print("Received execution result, updating history for next turn.")
            self._history.append(HumanMessage(content=input_text))

        else:
            print(f"Received unknown message type: {input_dict.get('kind')}")

        turn_info = f"\nCurrent turn: {self._turn_count + 1} of {self._max_turn_count}."                                                                                                                                                                                                   
        messages = [SystemMessage(content=SYSTEM_PROMPT + turn_info)] + self._history     

        last_error: terminal_bench_format_exception | None = None
        for attempt in range(self._max_syntax_retries):
            result = await self._invoke_llm_async(messages)
            response_text = getattr(result, "content", str(result))
            print(f"LLM response (attempt {attempt + 1}): {response_text}")

            try:
                response_text = self.postprocess_response(response_text, updater)
                self._history.append(AIMessage(content=response_text))
                return response_text
            except terminal_bench_format_exception as e:
                last_error = e
                print(f"Format error on attempt {attempt + 1}: {e.message}")
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content=json.dumps({
                    "kind": "error",
                    "error": e.message,
                })))

        raise last_error

    def _check_no_interactive_commands(self, command: str) -> None:
        tokens = command.strip().split()
        first_token = tokens[0].split("/")[-1]
        if first_token in self._ALWAYS_INTERACTIVE:
            raise terminal_bench_format_exception(
                f"Interactive command not allowed: {first_token!r}. "
                "Use non-interactive alternatives (e.g. python -c, git --no-pager)."
            )
        if first_token in self._REPL_COMMANDS and len(tokens) == 1:
            raise terminal_bench_format_exception(
                f"Bare REPL invocation not allowed: {first_token!r}. "
                f"Use {first_token} -c '...' or {first_token} script.py instead."
            )

    def _check_no_destructive_commands(self, command: str) -> None:
        for pattern in self._DESTRUCTIVE_PATTERNS:
            if re.search(pattern, command):
                raise terminal_bench_format_exception(
                    f"Potentially destructive command blocked: {command!r}"
                )

    def _check_command_syntax(self, command: str) -> None:
        if not command:
            raise terminal_bench_format_exception("exec_request has an empty command")
        self._check_no_interactive_commands(command)
        self._check_no_destructive_commands(command)
        result = subprocess.run(
            ["bash", "-n"],
            input=command,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise terminal_bench_format_exception(
                f"Command has invalid shell syntax: {result.stderr.strip()!r} — command was: {command!r}"
            )

    def postprocess_response(self, response_text: str, updater: TaskUpdater) -> str:
        """Post-process the LLM response to ensure it's valid JSON with expected structure."""

        # here we should have some sanity checks to ensure the response is
        # valid JSON with the expected structure and the command looks
        # reasonable (not rm -rf / or something) before
        # we send it back to the A2A server

        try:
            response_dict = json.loads(response_text)
        except json.JSONDecodeError:
            raise terminal_bench_format_exception(
                "LLM response is not valid JSON: " + response_text)

        if response_dict.get("kind") == "exec_request":
            command = response_dict.get("command", "")
            self._check_command_syntax(command)
            timeout = response_dict.get("timeout", 30)
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise terminal_bench_format_exception(
                    f"exec_request has invalid timeout: {timeout!r} (must be > 0)"
                )
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
