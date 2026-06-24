from a2a.server.tasks import TaskUpdater
from a2a.types import Message

from agents.configuration.config import (
    LLM_PROVIDER,
    SYSTEM_PROMPT,
    RECON_CMD,
    L3S_MODEL,
    L3S_ENDPOINT,
    ACADEMICCLOUD_MODEL,
    ACADEMICCLOUD_ENDPOINT,
    ENABLE_RATE_LIMIT_BACKOFF,
    BACKOFF_MAX_RETRIES,
    BACKOFF_BASE_DELAY,
    MAX_TURN_COUNT,
    MAX_SYNTAX_RETRIES,
    MAX_PLAN_TURNS,
    SHORT_TERM_WINDOW,
)
from agents.llm_clients.l3s import L3SLLMClient
from agents.llm_clients.academic_cloud import AcademicCloudLLMClient
from agents.llm_clients.open_router import OpenRouterLLMClient
from agents.llm_clients.llm_client import TerminalBenchLLMClientInterface
from agents.terminal_bench import TerminalBenchAgent


def _build_llm_client() -> TerminalBenchLLMClientInterface:
    if LLM_PROVIDER == "l3s":
        return L3SLLMClient(model=L3S_MODEL, base_url=L3S_ENDPOINT)
    if LLM_PROVIDER == "academiccloud":
        return AcademicCloudLLMClient(
            model=ACADEMICCLOUD_MODEL,
            base_url=ACADEMICCLOUD_ENDPOINT,
            backoff_enabled=ENABLE_RATE_LIMIT_BACKOFF,
            backoff_max_retries=BACKOFF_MAX_RETRIES,
            backoff_base_delay=BACKOFF_BASE_DELAY,
        )
    if LLM_PROVIDER == "openrouter":
        return OpenRouterLLMClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


class Agent:
    def __init__(self):
        self._backend = TerminalBenchAgent(
            llm_client=_build_llm_client(),
            system_prompt=SYSTEM_PROMPT,
            recon_cmd=RECON_CMD,
            max_turn_count=MAX_TURN_COUNT,
            max_syntax_retries=MAX_SYNTAX_RETRIES,
            max_plan_turns=MAX_PLAN_TURNS,
            short_term_window=SHORT_TERM_WINDOW,
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await self._backend.run(message, updater)
