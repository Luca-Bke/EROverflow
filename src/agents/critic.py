"""LLM-based format checker for the terminal agent.

Engaged by the Actor only after the static syntax checker
(`ResponseFormatChecker` + `ExecRequestChecker`) has failed once. Once engaged,
the CheckerAgent becomes the gate that approves a response for sending: it
givesthe Actor actionable feedback on format problems (most importantly:
emitting multiple commands in a single response) and decides when a corrected
response may be sent.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, override

from langchain_core.messages import BaseMessage, HumanMessage
from langsmith import traceable

from agents.abstract_agent import AbstractAgent
from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.terminal_bench_supplementary.terminal_bench_format_exception import (
    terminal_bench_format_exception,
)
from agents.tools.exec_request_checker import ExecRequestChecker
from agents.tools.response_format_checker import ResponseFormatChecker


@dataclass
class CriticVerdict:
    approved: bool
    feedback: str
    is_valid_verdict: bool = True
    error: bool = False  # True when the checker LLM itself was unavailable


class CriticAgent(AbstractAgent):
    """Second-stage, LLM-based validator and send-gate for the Actor."""

    def __init__(self, llm_client: AbstractLLMClient) -> None:
        super().__init__(llm_client)
        self._max_verdict_attempts = 10

    @staticmethod
    @traceable(name="ParseCriticVerdict", run_type="parser")
    def _parse_verdict(raw_critic_verdict: str) -> CriticVerdict:
        data: dict[str, Any] | None = None
        try:
            data = json.loads(raw_critic_verdict)
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{.*\}", raw_critic_verdict or "", re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict) or "approved" not in data:
            return CriticVerdict(
                approved=False,
                feedback="",
                is_valid_verdict=False,
                error=False
            )

        approved = bool(data.get("approved"))
        feedback = str(data.get("feedback", "") or "")
        return CriticVerdict(approved=approved, feedback=feedback)

    @staticmethod
    def _validate_response(response_text: str) -> dict:
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
        else:
            raise terminal_bench_format_exception(
                "LLM response JSON missing 'kind' field or has unknown "
                f"kind: {response_text}"
            )

        return response_dict

    @staticmethod
    def _compose_critic_message(
            messages: list[BaseMessage],
            exec_request_candidate: str,
            static_syntax_validation_message: str) -> list[BaseMessage]:

        exec_request_wrapped = (
            "Hallo"
            f"{exec_request_candidate}"
        )
        syntax_check_wrapped = (
            "Hallo"
            f"{static_syntax_validation_message}"
        )

        messages.append(HumanMessage(content=exec_request_wrapped))
        messages.append(HumanMessage(content=syntax_check_wrapped))

        return messages

    async def _review(self, messages: list[BaseMessage],
                      exec_request_candidate: str,
                      static_syntax_validation_message: str) -> CriticVerdict:
        """Review the Actor's response. Returns a verdict with feedback.

        On any internal/LLM failure, returns a non-approving verdict that falls
        back to the raw syntax error so the Actor still gets useful feedback.
        """

        combined_critic_messages = self._compose_critic_message(
            messages, exec_request_candidate, static_syntax_validation_message)

        return await self._llm_client.invoke_async(combined_critic_messages)

    @override
    @traceable(name="Critic", run_type="chain")
    async def invoke(self, messages: list[BaseMessage],
                     exec_request_candidate: str) -> CriticVerdict:
        # ── Static syntax checker — always the first, cheap gate ──────────

        static_syntax_validation_message = (
            "No syntax error detected during"
            "static syntax validation"
        )

        try:
            self._validate_response(exec_request_candidate)
        except terminal_bench_format_exception as e:
            static_syntax_validation_message = e.message

        for i in range(self._max_verdict_attempts):
            try:
                response = await self._review(messages, exec_request_candidate,
                                              static_syntax_validation_message)
            except Exception as e:
                return CriticVerdict(
                    approved=False,
                    feedback=f"Format invalid (checker unavailable: {e}). "
                    "Respond with exactly one JSON object.",
                    is_valid_verdict=False,
                    error=True,
                )

            raw_critic_verdict = getattr(response, "content", str(response))
            verdict: CriticVerdict = self._parse_verdict(raw_critic_verdict)

            if static_syntax_validation_message != (
                "No syntax error detected during"
                "static syntax validation"
            ):
                verdict.approved = False

            if (verdict.is_valid_verdict or verdict.error):
                return verdict

        return CriticVerdict(False, f"""Critic did not generate conclusive
                             verdict in {self._max_verdict_attempts}
                             attempts.""", False, True)


__all__ = ["CriticAgent", "CriticVerdict"]
