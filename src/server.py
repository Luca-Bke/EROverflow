import argparse
import os
import uvicorn
from langsmith import traceable, tracing_context

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


@traceable(name="server_startup", run_type="chain")
def emit_startup_trace(host: str, port: int, card_url: str | None) -> dict[str, str]:
    return {
        "event": "server_startup",
        "service": "EROverflow Terminal Agent",
        "host": host,
        "port": str(port),
        "card_url": card_url or "",
        "tracing_forced_by_code": "true",
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Emit one startup trace as a clear lifecycle marker in LangSmith.
    with tracing_context(enabled=bool(os.getenv("LANGSMITH_API_KEY"))):
        emit_startup_trace(args.host, args.port, args.card_url)

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/

    skill = AgentSkill(
        id="terminal-bench-shell",
        name="Terminal Bench Shell",
        description="Solves command-line tasks via the terminal-bench-shell-v1 protocol",
        tags=["terminal", "shell", "cli"],
        examples=["Fix the failing tests in this repo", "Install dependencies and run the build"]
    )

    agent_card = AgentCard(
        name="EROverflow Terminal Agent",
        description="A2A purple agent for Terminal Bench 2.0 — solves hard, realistic command-line tasks",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
