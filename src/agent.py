from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from agents.openrouter import OpenRouterAgent


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self._backend = OpenRouterAgent()


    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)
        print(f"Agent received message: {input_text}")

        # Replace this example code with your agent logic

        # delegate from interface to backend agent implementation 
        # (e.g. OpenRouterAgent)
        await self._backend.run(message, updater)

        # await updater.update_status(
        #     TaskState.working, new_agent_text_message("Thinking...")
        # )

        # await updater.add_artifact(
        #     parts=[Part(root=TextPart(text=input_text))],
        #     name="Echo",
        # )

