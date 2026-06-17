from a2a.server.tasks import TaskUpdater
from a2a.types import Message

from agents.terminal_bench import TerminalBenchAgent


class Agent:
    def __init__(self):
        self._backend = TerminalBenchAgent()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await self._backend.run(message, updater)
