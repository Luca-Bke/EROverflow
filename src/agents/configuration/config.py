from agents.llm_clients.academic_cloud import AcademicCloudLLMClient
from agents.llm_clients.l3s import L3SLLMClient
from agents.llm_clients.llm_client import TerminalBenchLLMClientInterface
from agents.llm_clients.open_router import OpenRouterLLMClient

# ── Agent prompt and recon ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a terminal agent solving complex command-line tasks in a live shell environment.

You will receive messages as JSON. Respond ONLY with a SINGLE valid JSON object — one of:
  {"kind": "exec_request", "command": "<shell command>", "timeout": 300}
or to organise your work into a plan you will work through step by step:
  {"kind": "plan", "steps": ["step 1", "step 2", "..."]}
or when the task is complete:
  {"kind": "final"}

CRITICAL — exactly ONE JSON object per response:
- Respond with EXACTLY ONE JSON object. NEVER emit several JSON objects in one response.
- NEVER send a sequence of commands at once. Send only the FIRST command, then WAIT for its
  execution result before issuing the next one.
- Do not include any text outside the JSON object.
- Always end the task with {"kind": "final"} once it is complete — do not leave it hanging.

Planning:
- You may send {"kind": "plan", "steps": [...]} to record or update a plan. A plan is internal:
  it is NOT executed in the shell, it is stored and shown back to you so you can work through it
  step by step. After sending a plan, issue the first exec_request of that plan.

Format checking:
- A checker validates your response format. If your response is malformed (e.g. multiple commands
  in one response), it returns concrete feedback. Read it and reply with a single, corrected JSON object.

Workflow:
- Turn 0 is an automatic reconnaissance command (pwd/ls/find/git/tools). Read its output.
- Then send a {"kind": "plan", ...} grounded in what recon revealed, and work through it step by step.

Command Execution Rules:
- Never use interactive commands (vim, nano, less, ssh -t, top, htop, etc.)
- Always use non-interactive flags: apt-get -y, git --no-pager, python -c, etc.
- Bound the output of long/noisy commands so they do not flood your context:
    apt-get install -y X > /tmp/log 2>&1; tail -n 40 /tmp/log
    pip install --break-system-packages X 2>&1 | tail -3
- VERIFY before you finish: actually run the task's test/verification (e.g. the test harness,
  the smoke test, or a grep check) and confirm it passes BEFORE sending {"kind": "final"}.
- If a command fails, diagnose and try a different approach
- You can send a maximum of 30 commands total
- If the stderr and stdout of a command are not relevant to your further succeeding (e.g. the output of apt-get install update),
then pipe the output to null, the output does not clog up the history
- When possible, use filters to find the relevant information in log files or similar data

"""

# Head+tail budget (chars) for stdout/stderr of a single exec_result kept in memory.
MAX_OUTPUT_CHARS = 6000

# Fixed turn-0 reconnaissance: grounds every later decision in the real
# environment. Sent deterministically (no LLM call) on the first task message.
RECON_CMD = (
    "echo '=== PWD ===' && pwd && "
    "echo '=== LS ===' && ls -la && "
    "echo '=== FILES ===' && find . -maxdepth 2 -not -path '*/.*' -type f | sort | head -40 && "
    "echo '=== GIT ===' && (git log --oneline -5 2>/dev/null || echo '(no git)') && "
    "echo '=== TOOLS ===' && (which python3 pip git curl make 2>/dev/null | head -10 || true)"
)

# ── LLM provider selection ────────────────────────────────────────────────────

LLM_PROVIDER = "l3s"

LLM_PROVIDER_DICTIONARY: dict[str, type[TerminalBenchLLMClientInterface]] = {
    "openrouter": OpenRouterLLMClient,
    "academiccloud": AcademicCloudLLMClient,
    "l3s": L3SLLMClient,
}

# ── L3S / LLMHub ─────────────────────────────────────────────────────────────

L3S_MODEL = "ollama/qwen3.6:27b"
L3S_ENDPOINT = "https://inference.kbs.uni-hannover.de/v1"

# ── AcademicCloud ─────────────────────────────────────────────────────────────

ACADEMICCLOUD_MODEL = "qwen3.6-35b-a3b"
ACADEMICCLOUD_ENDPOINT = "https://chat-ai.academiccloud.de/v1"

# ── Rate-limit backoff (used by AcademicCloudLLMClient and L3SLLMClient) ─────

ENABLE_RATE_LIMIT_BACKOFF = True
BACKOFF_MAX_RETRIES = 4
BACKOFF_BASE_DELAY = 5.0

# ── Agent turn limits ─────────────────────────────────────────────────────────

MAX_TURN_COUNT = 30
MAX_SYNTAX_RETRIES = 5
MAX_PLAN_TURNS = 3
SHORT_TERM_WINDOW = 10
