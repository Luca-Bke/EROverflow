import httpx
import pytest
from src.messenger import Messenger


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL (default: http://localhost:9009)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark live tests as `integration` so they are easy to select/deselect.

    Anything in test_agent.py or under tests/custom/ talks to a running agent
    server. They still run by default but skip cleanly (via the `agent` fixture)
    when no server is reachable; use `-m "not integration"` to hide them entirely.
    """
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if "tests/custom/" in nodeid or "/test_agent.py" in nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def agent(request):
    """Agent URL fixture. Agent must be running before tests start."""
    url = request.config.getoption("--agent-url")

    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        if response.status_code != 200:
            pytest.skip(f"Agent at {url} returned status {response.status_code}")
    except Exception as e:
        pytest.skip(f"Could not connect to agent at {url}: {e}")

    return url
