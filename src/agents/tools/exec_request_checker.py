

# Always interactive — no non-interactive mode exists
import re
import subprocess

from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception


class ExecRequestChecker():
    _ALWAYS_INTERACTIVE = frozenset({
        "vim", "vi", "nvim", "nano", "emacs", "pico",
        "less", "more", "man",
        "top", "htop", "btop",
        "ssh",
        "mysql", "psql",
    })

    # Interactive only when called without arguments (bare REPL invocation)
    _REPL_COMMANDS = frozenset({"python", "python3", "node", "irb", "iex"})

    _DESTRUCTIVE_PATTERNS = [
        r"rm\s+-[^\s]*r[^\s]*\s+/",   # rm -rf /
        r"dd\s+.*of=/dev/[sh]d",       # dd onto block device
        r":\(\)\s*\{.*\}",             # fork bomb
        r"mkfs\.",                      # format filesystem
        r">\s*/dev/[sh]d",             # write directly to block device
    ]

    def _check_no_interactive_commands(command: str) -> None:
        tokens = command.strip().split()
        first_token = tokens[0].split("/")[-1]
        if first_token in ExecRequestChecker._ALWAYS_INTERACTIVE:
            raise terminal_bench_format_exception(
                f"Interactive command not allowed: {first_token!r}. "
                "Use non-interactive alternatives (e.g. python -c, git --no-pager)."
            )
        if first_token in ExecRequestChecker._REPL_COMMANDS and len(tokens) == 1:
            raise terminal_bench_format_exception(
                f"Bare REPL invocation not allowed: {first_token!r}. "
                f"Use {first_token} -c '...' or {first_token} script.py instead."
            )

    def _check_no_destructive_commands(command: str) -> None:
        for pattern in ExecRequestChecker._DESTRUCTIVE_PATTERNS:
            if re.search(pattern, command):
                raise terminal_bench_format_exception(
                    f"Potentially destructive command blocked: {command!r}"
                )

    def check_command_syntax(command: str) -> None:
        if not command:
            raise terminal_bench_format_exception(
                "exec_request has an empty command")
        ExecRequestChecker._check_no_interactive_commands(command)
        ExecRequestChecker._check_no_destructive_commands(command)
        result = subprocess.run(
            ["bash", "-n"],
            input=command,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise terminal_bench_format_exception(
                f"command in exec request has invalid shell syntax: {result.stderr.strip()!r} — command was: {command!r}"
            )

    def check_timeout(timeout):
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise terminal_bench_format_exception(
                f"exec_request has invalid timeout: {timeout!r} (must be > 0)"
            )

    def check_exec_request(request_dict: dict):
        command = request_dict.get("command", "")
        ExecRequestChecker.check_command_syntax(command)

        timeout = request_dict.get("timeout", 30)
        ExecRequestChecker.check_timeout(timeout)
