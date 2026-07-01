import json
import os
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import new_agent_text_message
from langsmith import tracing_context

from agents.configuration import config
from agents.llm_clients.l3s import L3SLLMClient
from agents.llm_clients.academic_cloud import AcademicCloudLLMClient
from agents.llm_clients.open_router import OpenRouterLLMClient
from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.terminal_bench import TerminalBenchAgent
from agents.terminal_bench_supplementary import utils


def _build_llm_client() -> AbstractLLMClient:
    if config.LLM_PROVIDER == "l3s":
        return L3SLLMClient(
            model=config.L3S_MODEL,
            base_url=config.L3S_ENDPOINT,
            timeout=config.L3S_REQUEST_TIMEOUT,
            backoff_enabled=config.ENABLE_RATE_LIMIT_BACKOFF,
            backoff_max_retries=config.BACKOFF_MAX_RETRIES,
            backoff_base_delay=config.BACKOFF_BASE_DELAY,
        )
    if config.LLM_PROVIDER == "academiccloud":
        return AcademicCloudLLMClient(
            model=config.ACADEMICCLOUD_MODEL,
            base_url=config.ACADEMICCLOUD_ENDPOINT,
            backoff_enabled=config.ENABLE_RATE_LIMIT_BACKOFF,
            backoff_max_retries=config.BACKOFF_MAX_RETRIES,
            backoff_base_delay=config.BACKOFF_BASE_DELAY,
        )
    if config.LLM_PROVIDER == "openrouter":
        return OpenRouterLLMClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}")


class Agent:
    def __init__(self):
        self._backend = TerminalBenchAgent(
            llm_client=_build_llm_client(),
            planner_system_prompt=config.PLANNER_SYSTEM_PROMPT,
            actor_system_prompt=config.ACTOR_SYSTEM_PROMPT,
            critic_system_prompt=config.CRITIC_SYSTEM_PROMPT,
            max_critic_actor_rounds=10,
            short_term_window=config.SHORT_TERM_WINDOW,
        )
        self._trace_enabled = bool(os.getenv("LANGSMITH_API_KEY"))
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "EROverflow-terminal-bench"
        if not os.getenv("LANGSMITH_ENDPOINT"):
            os.environ["LANGSMITH_ENDPOINT"] = "api.smith.langchain.com"
        self._turn_count = 0
        self._max_turn_count = config.MAX_TURN_COUNT
        self._memory_log: list[dict[str, Any]] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """ Check max turn count implement final tracing via langchain.
        Message processing is performed elsewhere. """
        # Out of turn budget: send a clean final so the executor never has to
        # respond with an empty message (which it treats as an error).
        if self._turn_count >= self._max_turn_count:
            print("Max turn count reached; sending final.")
            final_msg = json.dumps({"kind": "final"})
            await updater.complete(updater.new_agent_message(
                parts=[Part(root=TextPart(text=final_msg))]))
            return

        await updater.start_work(
            new_agent_text_message(f"Turn {self._turn_count}: thinking...")
        )

        response_result = await self._backend.handle_request_iteration(message, updater)

        # Defensive: the backend now always returns a JSON string or raises, so
        # a None here means a contract violation — fail loudly with a clear
        # message instead of the cryptic "json object ... not NoneType".
        if response_result is None:
            raise ValueError(
                "Backend returned no response (None) — see agent logs for the "
                "underlying cause."
            )

        self._memory_log.append(self._backend._memory.snapshot_memory())

        if (utils.is_final_response(response_result) or
                self._backend._llm_client.rate_limited()):

            with tracing_context(enabled=self._trace_enabled):
                utils.emit_session_trace(
                    history=self._memory_log,
                    turn_count=self._turn_count,
                    rate_limited=self._backend._llm_client.rate_limited(),
                    retry_log=self._backend._llm_client.retry_log(),
                )

        # Send the agent response back to the A2A server. A task lives for only
        # one turn, so we must complete it exactly once — otherwise the executor
        # responds with an empty message and errors.
        response_msg = updater.new_agent_message(
            parts=[Part(root=TextPart(text=response_result))]
        )
        await updater.complete(response_msg)

        self._turn_count += 1
