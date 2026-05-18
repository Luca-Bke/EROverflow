import asyncio
import sys
sys.path.insert(0, 'src')
from messenger import Messenger

async def main():
    m = Messenger()
    response = await m.talk_to_agent(
        message="My wrist hurts",
        url="http://192.168.178.21:9009",
        new_conversation=True
    )
    print(response)

asyncio.run(main())