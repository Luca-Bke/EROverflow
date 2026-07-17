"""Tool and schema definitions for the Actor-Critic pipeline.

The Actor uses native LLM tool calls instead of emitting JSON strings.
This module holds the declarative tool definitions and JSON schemas used
for structured outputs (e.g. Critic verdicts).
"""

from __future__ import annotations

# ── Actor tool definitions ───────────────────────────────────────────────────

# OpenAI-compatible tool definition for the execute_command tool.
# The Actor receives this list of tools and can invoke `execute_command`.
EXECUTE_COMMAND_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": (
            "Execute a single shell command in the live environment. "
            "The command will be reviewed by the Critic for safety before execution. "
            "Use this tool to run one command at a time and wait for the result "
            "before issuing the next command."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The shell command to execute. Must be non-interactive "
                        "and non-destructive. Examples: 'cat /etc/passwd', "
                        "'grep -r \"flag\" .', 'python3 solve.py'"
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Maximum execution time in seconds (1-300). "
                        "Default is 300 if omitted."
                    ),
                    "minimum": 1,
                    "maximum": 300,
                    "default": 300,
                },
            },
            "required": ["command"],
        },
    },
}

# OpenAI-compatible tool definition for the submit_final tool.
# The Actor calls this to signal task completion.
SUBMIT_FINAL_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_final",
        "description": (
            "Submit the final result and end the task. "
            "Call this only when the task is complete and verified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished.",
                },
            },
            "required": ["output"],
            "additionalProperties": False,
        },
    },
}

# List of tools the Actor can call.
ACTOR_TOOLS: list[dict] = [EXECUTE_COMMAND_TOOL, SUBMIT_FINAL_TOOL]

# ── Critic verdict JSON schema (for response_format) ─────────────────────────

# JSON Schema used as the `response_format` for the Critic's LLM call.
# Guarantees structured output: {"approved": bool, "feedback": str}
CRITIC_VERDICT_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "critic_verdict",
        "schema": {
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": (
                        "True if the command is safe and structurally valid "
                        "and should be executed. False if it should be rejected."
                    ),
                },
                "feedback": {
                    "type": "string",
                    "description": (
                        "When approved=False: one actionable instruction for the "
                        "Actor to fix the command. When approved=True: must be empty."
                    ),
                },
            },
            "required": ["approved", "feedback"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT = 300
MAX_TIMEOUT = 300


def clamp_timeout(timeout: int | None) -> int:
    """Normalise and clamp a timeout value to [1, MAX_TIMEOUT]."""
    if timeout is None:
        return DEFAULT_TIMEOUT
    return min(max(int(timeout), 1), MAX_TIMEOUT)


__all__ = [
    "ACTOR_TOOLS",
    "CRITIC_VERDICT_SCHEMA",
    "EXECUTE_COMMAND_TOOL",
    "SUBMIT_FINAL_TOOL",
    "clamp_timeout",
    "DEFAULT_TIMEOUT",
    "MAX_TIMEOUT",
]