from a2a.server.tasks import TaskUpdater
from a2a.types import Message

from agents.terminal_bench import TerminalBenchAgent
from langsmith import traceable, tracing_context


class Agent:
    def __init__(self):
        self._backend = TerminalBenchAgent()

    @traceable
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await self._backend.run(message, updater)