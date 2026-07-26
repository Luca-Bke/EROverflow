"""Critic agent that validates tool-call commands before execution.

Receives a tool call (command + timeout) from the Actor, runs a static
syntax/safety check, then delegates to an LLM reviewer with structured
output (response_format JSON schema). The verdict determines whether the
command is approved for execution or rejected with feedback.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langsmith import traceable

from agents.abstract_agent import AbstractAgent
from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.llm_clients.retry import is_retryable
from agents.terminal_bench_supplementary import utils
from agents.terminal_bench_supplementary.terminal_bench_format_exception import (
    terminal_bench_format_exception,
)
from agents.terminal_bench_supplementary.utils import TimeTracer
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.tool_definitions import CRITIC_VERDICT_SCHEMA


logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    approved: bool
    feedback: str
    is_valid_verdict: bool = True
    error: bool = False  # True when the checker LLM itself was unavailable


class CriticAgent(AbstractAgent):
    """Safety gate for the Actor's tool calls.

    Validates commands via static checks (syntax, interactive, destructive)
    and LLM-based review. Uses structured output (response_format) for
    reliable verdict parsing.
    """

    def __init__(self, llm_client: AbstractLLMClient) -> None:
        super().__init__(llm_client)
        self._max_verdict_attempts = 10

    @staticmethod
    @traceable(name="ParseCriticVerdict", run_type="parser")
    @TimeTracer.timed("CriticAgent._parse_verdict")
    def _parse_verdict(raw_critic_verdict: str) -> CriticVerdict:
        """Parse the LLM's structured verdict response.

        Handles edge cases like double braces ({{...}}), embedded prose,
        and malformed JSON.
        """
        text = (raw_critic_verdict or "").strip()
        data: dict[str, Any] | None = None

        # Remove excessive outer braces until we have exactly one pair
        # LLMs sometimes emit {{...}}, {{{...}}}, or even asymmetric {{...}
        while text.startswith("{") and text.endswith("}"):
            # Count leading and trailing braces
            leading = 0
            for ch in text:
                if ch == "{":
                    leading += 1
                else:
                    break
            trailing = 0
            for ch in reversed(text):
                if ch == "}":
                    trailing += 1
                else:
                    break

            # If excessive braces on either side, strip down to one pair
            if leading > 1 or trailing > 1:
                # Remove all but one outer brace pair
                text = text[leading - 1 : len(text) - trailing + 1].strip()
                continue

            # Exactly one pair – try to parse as JSON
            try:
                candidate = json.loads(text)
                if isinstance(candidate, dict):
                    data = candidate
                    break  # valid JSON — use it
            except json.JSONDecodeError:
                # Not valid JSON with one pair, can't strip further
                break

        # Fallback: try raw_decode on the original
        if data is None:
            try:
                candidate = json.loads(text)
                if isinstance(candidate, dict):
                    data = candidate
            except (json.JSONDecodeError, TypeError):
                m = re.search(r"\{.*\}", raw_critic_verdict, re.DOTALL)
                if m:
                    try:
                        candidate = json.loads(m.group(0))
                        if isinstance(candidate, dict):
                            data = candidate
                    except json.JSONDecodeError:
                        pass

        if not isinstance(data, dict) or "approved" not in data:
            return CriticVerdict(
                approved=False,
                feedback="",
                is_valid_verdict=False,
                error=False,
            )

        approved = bool(data.get("approved"))
        feedback = str(data.get("feedback", "") or "")
        return CriticVerdict(approved=approved, feedback=feedback)

    @staticmethod
    def _validate_command(command: str) -> None:
        """Static syntax + safety validation on the command string.

        Checks shell syntax, interactive commands, and destructive patterns.
        Raises terminal_bench_format_exception on any issue.
        """
        req = {"command": command, "timeout": 300}
        ExecRequestChecker.check_exec_request(req)

    @staticmethod
    def _compose_critic_messages(
        messages: list[BaseMessage],
        command: str,
        timeout: int,
        static_syntax_validation_message: str,
    ) -> list[BaseMessage]:
        """Build the message list for the Critic LLM call."""
        # Present the tool call arguments as a human-readable message
        tool_call_info = HumanMessage(
            content=(
                f"Actor wants to execute the following command:\n"
                f"  command: {command!r}\n"
                f"  timeout: {timeout}"
            )
        )
        messages.append(tool_call_info)

        syntax_check_wrapped = utils.apply_message_label(
            HumanMessage(content=static_syntax_validation_message),
            "static_syntax_validation_message",
        )
        messages.append(syntax_check_wrapped)

        return messages

    @traceable(name="Critic", run_type="chain")
    async def invoke(
        self,
        messages: list[BaseMessage],
        tool_call: dict[str, Any],
    ) -> CriticVerdict:
        """Review a tool call from the Actor.

        Args:
            messages: Base messages (system prompt, etc.).\n
            tool_call: Dict with 'name', 'args' (dict with 'command',
                       'timeout'), and optionally 'id'.
        """
        # LangChain uses 'args'; raw OpenAI uses 'arguments' — handle both
        arguments: dict[str, Any] = tool_call.get("args") or tool_call.get("arguments", {})
        # Ensure arguments is a dict (some LLMs return JSON strings)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        command: str = arguments.get("command", "")
        timeout: int = arguments.get("timeout", 300)

        # ── Static syntax checker — always the first, cheap gate ──────────
        static_syntax_validation_message = (
            "No syntax error detected during static syntax validation"
        )

        try:
            self._validate_command(command)
        except terminal_bench_format_exception as e:
            static_syntax_validation_message = e.message

        for _ in range(self._max_verdict_attempts):
            try:
                critic_messages = self._compose_critic_messages(
                    list(messages),  # don't mutate the original
                    command,
                    timeout,
                    static_syntax_validation_message,
                )

                try:
                    response = await self._llm_client.invoke_with_response_format_async(
                        critic_messages, CRITIC_VERDICT_SCHEMA
                    )
                except Exception as exc:
                    if not is_retryable(exc):
                        # Non-retryable error — give up and return error verdict
                        return CriticVerdict(
                            approved=False,
                            feedback=f"Format invalid (checker unavailable: {exc}). "
                            "Try a simpler command.",
                            is_valid_verdict=False,
                            error=True,
                        )
                    # Retryable error — retry immediately (separate LangSmith trace)
                    logger.warning(
                        "Critic LLM call failed (%s), retrying immediately...",
                        type(exc).__name__,
                    )
                    response = await self._llm_client.invoke_with_response_format_async(
                        critic_messages, CRITIC_VERDICT_SCHEMA
                    )
            except Exception as e:
                return CriticVerdict(
                    approved=False,
                    feedback=f"Format invalid (checker unavailable: {e}). "
                    "Try a simpler command.",
                    is_valid_verdict=False,
                    error=True,
                )

            raw_critic_verdict = getattr(response, "content", str(response))
            verdict: CriticVerdict = self._parse_verdict(raw_critic_verdict)

            # Static check failure always overrides LLM approval
            if static_syntax_validation_message != (
                "No syntax error detected during static syntax validation"
            ):
                verdict.approved = False
                # Prepend the static error if the LLM didn't catch it
                if verdict.feedback:
                    verdict.feedback = (
                        f"Static check: {static_syntax_validation_message}. "
                        f"LLM feedback: {verdict.feedback}"
                    )
                else:
                    verdict.feedback = static_syntax_validation_message

            if verdict.is_valid_verdict or verdict.error:
                return verdict

        return CriticVerdict(
            False,
            f"Critic did not generate conclusive verdict in "
            f"{self._max_verdict_attempts} attempts.",
            False,
            True,
        )


__all__ = ["CriticAgent", "CriticVerdict"]
