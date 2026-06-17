"""LLM-based format checker for the terminal agent.

Engaged by the Actor only after the static syntax checker
(`ResponseFormatChecker` + `ExecRequestChecker`) has failed once. Once engaged,
the CheckerAgent becomes the gate that approves a response for sending: it gives
the Actor actionable feedback on format problems (most importantly: emitting
multiple commands in a single response) and decides when a corrected response
may be sent.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


CHECKER_SYSTEM_PROMPT = """\
You are a strict format checker for a terminal agent (the "Actor").

The Actor must respond with EXACTLY ONE JSON object, one of:
  {"kind": "exec_request", "command": "<shell command>", "timeout": 300}
  {"kind": "final"}
  {"kind": "plan", "steps": ["step 1", "step 2", ...]}

The single most common mistake you must catch: the Actor emits SEVERAL JSON
objects in one response (e.g. a whole sequence of exec_request objects, or
exec_request objects followed by a final). This is INVALID — only one object,
one command, per turn is allowed. The Actor must send only the first command and
wait for its execution result before issuing the next.

You receive the Actor's raw response and, when present, the syntax error the
static checker produced. Your job:
- If a syntax error is given: diagnose the real cause and give ONE precise,
  actionable instruction to fix it. If multiple commands were emitted, say so
  explicitly and tell the Actor to send only the first single command.
- If no syntax error is given: you are re-validating a corrected response.
  Decide whether it is now a single, valid JSON object of the allowed kinds.

Respond with ONLY a JSON object, no surrounding text:
  {"approved": true, "feedback": ""}
or
  {"approved": false, "feedback": "<one concrete instruction to the Actor>"}
"""


@dataclass
class CheckVerdict:
    approved: bool
    feedback: str
    error: bool = False  # True when the checker LLM itself was unavailable


class CheckerAgent:
    """Second-stage, LLM-based validator and send-gate for the Actor."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.getenv(
            "CHECKER_MODEL",
            os.getenv("ACADEMICCLOUD_MODEL", "qwen3-coder-30b-a3b-instruct"),
        )
        self._base_url = os.getenv(
            "ACADEMICCLOUD_ENDPOINT", "https://chat-ai.academiccloud.de/v1"
        )
        self._temperature = float(os.getenv("CHECKER_TEMPERATURE", "0.0"))
        self._llm: ChatOpenAI | None = None

    def _get_api_key(self) -> str:
        api_key = os.getenv("ACADEMICCLOUD_API_KEY")
        if not api_key:
            raise ValueError("ACADEMICCLOUD_API_KEY environment variable not set")
        return api_key

    def _create_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self._model,
                api_key=self._get_api_key(),
                base_url=self._base_url,
                temperature=self._temperature,
                tags=["eroverflow", "terminal-bench", "checker"],
                metadata={"agent": "checker", "provider": "academiccloud"},
            )
        return self._llm

    async def review(self, actor_response: str, syntax_error: str | None) -> CheckVerdict:
        """Review the Actor's response. Returns a verdict with feedback.

        On any internal/LLM failure, returns a non-approving verdict that falls
        back to the raw syntax error so the Actor still gets useful feedback.
        """
        user_payload = {
            "actor_response": actor_response,
            "syntax_error": syntax_error,
        }
        messages = [
            SystemMessage(content=CHECKER_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(user_payload)),
        ]

        try:
            llm = self._create_llm()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: llm.invoke(messages))
            raw = getattr(result, "content", str(result))
            return self._parse_verdict(raw, syntax_error)
        except Exception as e:
            return CheckVerdict(
                approved=False,
                feedback=syntax_error
                or f"Format invalid (checker unavailable: {e}). "
                "Respond with exactly one JSON object.",
                error=True,
            )

    @staticmethod
    def _parse_verdict(raw: str, syntax_error: str | None) -> CheckVerdict:
        data: dict[str, Any] | None = None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict) or "approved" not in data:
            # Could not parse a verdict — do not approve, surface what we have.
            return CheckVerdict(
                approved=False,
                feedback=(raw or syntax_error or "Respond with exactly one JSON object.").strip(),
            )

        approved = bool(data.get("approved"))
        feedback = str(data.get("feedback", "") or "")
        return CheckVerdict(approved=approved, feedback=feedback)


__all__ = ["CheckerAgent", "CheckVerdict"]
