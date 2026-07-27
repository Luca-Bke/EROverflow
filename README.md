# EROverflow Terminal Agent

An A2A (Agent-to-Agent) **purple agent** for **Terminal Bench 2.0**. It solves
hard, realistic command-line tasks by issuing shell commands one at a time over
the [`terminal-bench-shell-v1`](https://a2a-protocol.org/latest/) protocol,
guarded by a safety-reviewing critic.

The agent is built around a **Planner → Actor → Critic → Shell** pipeline: a
Planner drafts a step-by-step plan, an Actor proposes the next shell command as a
native LLM tool call, and a Critic reviews every command for safety before it is
executed by the environment.

## Architecture

```
A2A request
    │
    ▼
server.py ──► Executor ──► Agent ──► TerminalBenchAgent  (orchestrator)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                      ▼                      ▼
               PlannerAgent           ActorAgent             CriticAgent
          plan + task framing   proposes tool calls    approves / rejects
                                (execute_command,       each command before
                                 submit_final)          it reaches the shell
```

- **Planner** (`src/agents/planner.py`) — runs on the first turn; produces a
  structured plan `{updated_plan, task_formulation}` from the task.
- **Actor** (`src/agents/actor.py`) — drives the work using native LLM tool
  calls: `execute_command` to run a shell command, `submit_final` when the whole
  task is done.
- **Critic** (`src/agents/critic.py`) — a safety gate that runs as middleware.
  It intercepts every proposed command, applies static + LLM checks, and returns
  a `CriticVerdict` that either approves the command (it is executed) or rejects
  it with feedback (the Actor retries).
- **Orchestrator** (`src/agents/terminal_bench.py`, `TerminalBenchAgent`) — runs
  the Actor–Critic loop each turn (up to `max_critic_actor_rounds = 10`) and
  implements the `terminal-bench-shell-v1` message protocol.

The A2A entry chain is `server.py` → `Executor` (`src/executor.py`, one `Agent`
instance per A2A context) → `Agent` (`src/agent.py`) → `TerminalBenchAgent`.

## Project structure

```
src/
├─ server.py                 # A2A server + agent-card definition (entry point)
├─ executor.py               # A2A request handler; per-context Agent instances
├─ agent.py                  # Agent wrapper: builds the LLM client, wires tracing
├─ messenger.py              # A2A messaging utilities
└─ agents/
   ├─ planner.py             # PlannerAgent
   ├─ actor.py               # ActorAgent (tool calls)
   ├─ critic.py              # CriticAgent (safety gate)
   ├─ terminal_bench.py      # TerminalBenchAgent orchestrator
   ├─ abstract_agent.py
   ├─ configuration/
   │  └─ config.py           # system prompts + all model/endpoint/limit config
   ├─ llm_clients/           # l3s / academic_cloud / open_router clients + retry
   ├─ tools/                 # agent_memory, exec/response checkers, tool schemas
   └─ terminal_bench_supplementary/   # utils (TimeTracer), pipeline messages, exceptions
tests/                       # unit + integration tests
Dockerfile                   # container image (uv-based)
amber-manifest.json5         # Amber deployment manifest
```

## LLM providers

Three providers are supported. The active one is selected by the **source-code
constant** `LLM_PROVIDER` in `src/agents/configuration/config.py` — this is *not*
an environment variable; change it in code to switch providers. Only the
selected provider's API key needs to be set.

| Provider (`LLM_PROVIDER`) | Model | Endpoint | API-key env var |
|---|---|---|---|
| `l3s` *(default)* | `vllm/qwen3.6:35b-a3b-bf16` | `https://inference.kbs.uni-hannover.de/v1` | `LLMHUB_APIKEY` |
| `academiccloud` | `qwen3.6-35b-a3b` | `https://chat-ai.academiccloud.de/v1` | `ACADEMICCLOUD_API_KEY` |
| `openrouter` | `qwen/qwen3.6-27b` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |

Models, endpoints, and timeouts live in `config.py`; the client implementations
are in `src/agents/llm_clients/`.

## Configuration

### Environment (`.env`)

Copy `.env.example` to `.env` and fill in the values you need:

```bash
cp .env.example .env
```

It documents:

- **API keys** — `LLMHUB_APIKEY`, `ACADEMICCLOUD_API_KEY`, `OPENROUTER_API_KEY`
  (only the active provider's key is required).
- **LangSmith tracing** — `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`,
  `LANGSMITH_ENDPOINT`, `LANGSMITH_TRACING`.
- **Logging** — `LOG_LEVEL` (see [Logging](#logging)).

Rate-limit backoff (`ENABLE_RATE_LIMIT_BACKOFF`, `BACKOFF_MAX_RETRIES`,
`BACKOFF_BASE_DELAY`) and the provider models/endpoints are source-code
constants in `config.py`, not environment variables.

### Tuning knobs (`config.py`)

Behavioural limits live in `src/agents/configuration/config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `MAX_TURN_COUNT` | 60 | Max A2A turns before the agent finalizes |
| `MAX_PLAN_TURNS` | 3 | Planner invocations budget |
| `SHORT_TERM_WINDOW` | 10 | Rolling conversation window kept in memory |
| `MAX_SYNTAX_RETRIES` | 5 | Critic/format retry budget |
| `MAX_OUTPUT_CHARS` | 6000 | Truncation bound for command output |

## Running locally

```bash
# Install dependencies
uv sync

# Run the server (loads variables from .env)
uv run --env-file .env src/server.py
```

By default the server binds to **`127.0.0.1:9010`**. Flags (`src/server.py`):

- `--host` (default `127.0.0.1`)
- `--port` (default `9010`)
- `--card-url` (URL advertised in the agent card; optional)

The agent card is then served at
`http://127.0.0.1:9010/.well-known/agent-card.json`.

## Logging

Runtime diagnostics use Python's `logging` module, written to **stderr** and
controlled by the `LOG_LEVEL` environment variable:

- `INFO` *(default)* — shows the routine decision-trace (approved/rejected
  commands, finalization, loop exhaustion).
- `WARNING` — quiets the trace to anomalies and errors only.
- `DEBUG` — most verbose.

```bash
LOG_LEVEL=WARNING uv run --env-file .env src/server.py
```

## Tracing (LangSmith)

Tracing is enabled automatically when `LANGSMITH_API_KEY` is set. If
`LANGSMITH_PROJECT` is not provided, it defaults to `EROverflow-terminal-bench`
(`src/agent.py`). Set the variables in your `.env`:

```dotenv
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT=EROverflow-terminal-bench
# optional, e.g. for the EU region or a self-hosted instance:
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

Or export them in your shell before launching (bash/zsh):

```bash
export LANGSMITH_API_KEY="<your-langsmith-api-key>"
```

## Running with Docker

```bash
# Build the image
docker build -t eroverflow .

# Run the container (serves 0.0.0.0:9010 inside the container)
docker run -p 9010:9010 --env-file .env eroverflow
```

The image entrypoint runs `uv run src/server.py --host 0.0.0.0 --port 9010`.

## Testing

```bash
# Install test dependencies
uv sync --extra test
```

The suite splits into offline **unit** tests and **integration** tests (marked
`integration`, they need a running agent and skip cleanly when none is reachable).

```bash
# Fast, offline unit tests only — no server required
uv run pytest -m "not integration"

# Full suite against a running agent (start it first, see "Running locally")
uv run pytest --agent-url http://localhost:9010
```

Integration and debug tests live in `tests/custom/` (e.g. `test_debug_planner.py`,
`test_terminal_bench_loop.py`) and can be run individually with
`--capture=no` to see their output. Note: `tests/conftest.py` defaults
`--agent-url` to `http://localhost:9009`, so pass `--agent-url` explicitly to
match whichever port your agent is actually serving.

## Deployment / Publishing

The agent is deployed via the Amber manifest (`amber-manifest.json5`), which
points at the image `ghcr.io/luca-bke/eroverflow` and serves the A2A endpoint on
port 9010.

CI is defined in `.github/workflows/test-and-publish.yml`. On each push it:

1. Builds the Docker image.
2. Runs the container and waits for the agent card.
3. Runs the test suite against it. *(CI runs the container on port 9009
   internally.)*
4. On success (non-PR), publishes to GitHub Container Registry — the `latest`
   tag on `main`, plus semantic-version tags for `v*` git tags.
5. On `main` only, dispatches the downstream leaderboard workflow.

If the agent needs secrets in CI, add them under **Settings → Secrets and
variables → Actions → Repository secrets**; they are exposed as environment
variables during the test step.
