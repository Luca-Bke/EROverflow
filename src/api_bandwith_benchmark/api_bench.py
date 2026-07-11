"""LLM endpoint latency/bandwidth benchmark.

The multi-agent pipeline occasionally sees wildly inconsistent answer times
(e.g. 0.3s, 0.5s, 229.0s for otherwise similar calls), which breaks turn-based
orchestration. This script hammers the configured LLM endpoint through the
same `AbstractLLMClient` interface the agents use, across three prompt-length
categories, to check whether the spikes are endpoint-side noise or correlate
with prompt length.

Run directly: `python src/api_bandwith_benchmark/api_bench.py [options]`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

# Make the `agents` package (src/agents/...) importable when this file is
# run directly, without relying on external PYTHONPATH configuration.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib.pyplot as plt  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from agents.configuration import config  # noqa: E402
from agents.llm_clients.abstract_llm_client import AbstractLLMClient  # noqa: E402
from agents.tools.agent_memory import AgentMemory  # noqa: E402

DEFAULT_NUM_REQUESTS = 50

CATEGORY_LABELS = {
    "single_prompt": "Single prompt",
    "multi_prompt": "Multi prompt",
    "long_context_prompt": "Long context prompt",
}

# Fixed categorical order, matches the identity of each prompt category.
CATEGORY_COLORS = {
    "single_prompt": "#2a78d6",        # blue
    "multi_prompt": "#1baf7a",         # aqua
    "long_context_prompt": "#eda100",  # yellow
}


# ── Environment setup ─────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """Populate os.environ from the repo-root .env file, if present.

    Avoids adding a python-dotenv dependency just for this script; only sets
    keys that aren't already present in the environment.
    """
    env_path = _SRC_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ── Prompt builders ────────────────────────────────────────────────────────────
# The multi- and long-context categories are built with AgentMemory, the same
# class the planner/actor/critic agents use to assemble their prompts.

def build_single_prompt() -> list[Any]:
    """A single, short human message — the baseline case."""
    return [HumanMessage(content="In one short sentence, confirm you are online and ready.")]


def build_multi_prompt() -> list[Any]:
    """A realistic multi-turn actor prompt: system + task + plan + short-term history."""
    memory = AgentMemory(
        planner_system_prompt=SystemMessage(
            content=config.PLANNER_SYSTEM_PROMPT),
        actor_system_prompt=SystemMessage(content=config.ACTOR_SYSTEM_PROMPT),
        critic_system_prompt=SystemMessage(
            content=config.CRITIC_SYSTEM_PROMPT),
        short_term_window=config.SHORT_TERM_WINDOW,
    )
    memory.set_task(
        "Locate and read the configuration file for the deployed service.")
    memory.set_plan(
        "1) List the repo root. 2) Find config files. 3) Read the relevant one."
    )
    memory.set_subtask_formulation(
        "Run `ls -la` in the project root and report the output."
    )
    turns = [
        ("ls -la", "total 24\ndrwxr-xr-x  5 user user 4096 Jul 11 10:00 .\n"
         "-rw-r--r--  1 user user  512 Jul 11 09:58 Makefile\n"
         "-rw-r--r--  1 user user 1240 Jul 11 09:58 README.md\n"
         "drwxr-xr-x  2 user user 4096 Jul 11 09:59 config"),
        ("find . -maxdepth 2 -iname '*.toml'",
         "./pyproject.toml\n./config/settings.toml"),
        ("cat config/settings.toml",
         "[server]\nhost = \"0.0.0.0\"\nport = 8080\n"),
        ("git log --oneline -5", "a1b2c3d fix request timeout\n9f8e7d6 add retry logic"),
    ]
    for cmd, output in turns:
        memory.add(HumanMessage(content=f"$ {cmd}\n{output}"))
    return memory.build_actor_messages()


def build_long_context_prompt(num_log_messages: int = 15) -> list[Any]:
    """A long actor prompt padded with many synthetic shell-log turns."""
    memory = AgentMemory(
        planner_system_prompt=SystemMessage(
            content=config.PLANNER_SYSTEM_PROMPT),
        actor_system_prompt=SystemMessage(content=config.ACTOR_SYSTEM_PROMPT),
        critic_system_prompt=SystemMessage(
            content=config.CRITIC_SYSTEM_PROMPT),
        short_term_window=num_log_messages,
    )
    memory.set_task(
        "Diagnose why the CI pipeline intermittently times out and propose a fix.")
    memory.set_plan(
        "1) Inspect CI logs. 2) Reproduce locally. 3) Bisect recent commits. "
        "4) Patch the flaky step. 5) Verify with a clean run."
    )
    memory.set_subtask_formulation(
        "Read through the last CI log excerpts and summarize recurring failure patterns."
    )
    log_line = (
        "2026-07-11T10:00:{:02d}Z INFO worker-{} processing job batch, "
        "queue_depth={} latency_ms={} retries={}\n"
    )
    for i in range(num_log_messages):
        block = "".join(
            log_line.format((i + j) % 60, i, 120 + j * 3, 80 + j * 7, j % 4)
            for j in range(25)
        )
        memory.add(HumanMessage(
            content=f"$ tail -n 40 /var/log/ci/job_{i}.log\n{block}"))
    return memory.build_actor_messages()


PROMPT_BUILDERS: dict[str, Callable[[], list[Any]]] = {
    "single_prompt": build_single_prompt,
    "multi_prompt": build_multi_prompt,
    "long_context_prompt": build_long_context_prompt,
}


# ── Benchmark execution ────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    category: str
    index: int
    prompt_chars: int
    elapsed_seconds: float
    success: bool
    error: str | None = None


async def run_trial(client: AbstractLLMClient, category: str, index: int, messages: list[Any]) -> TrialResult:
    prompt_chars = sum(len(str(m.content)) for m in messages)
    start = time.perf_counter()
    try:
        await client.invoke_async(messages)
        elapsed = time.perf_counter() - start
        return TrialResult(category, index, prompt_chars, elapsed, True)
    except Exception as e:  # noqa: BLE001 - we want to record and continue past any failure
        elapsed = time.perf_counter() - start
        return TrialResult(category, index, prompt_chars, elapsed, False, str(e)[:300])


async def run_benchmark(
    client: AbstractLLMClient, num_requests: int, delay_seconds: float
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for category, builder in PROMPT_BUILDERS.items():
        print(
            f"\n=== {CATEGORY_LABELS[category]} ({num_requests} requests) ===")
        for i in range(num_requests):
            messages = builder()
            result = await run_trial(client, category, i, messages)
            results.append(result)
            status = "ok" if result.success else f"ERROR: {result.error}"
            print(
                f"  [{i + 1}/{num_requests}] {result.elapsed_seconds:6.2f}s "
                f"({result.prompt_chars} chars) {status}"
            )
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def summarize(results: list[TrialResult]) -> dict[str, dict[str, float]]:
    by_category: dict[str, list[float]] = defaultdict(list)
    errors_by_category: dict[str, int] = defaultdict(int)
    for r in results:
        by_category[r.category].append(r.elapsed_seconds)
        if not r.success:
            errors_by_category[r.category] += 1

    summary = {}
    for category, times in by_category.items():
        sorted_times = sorted(times)
        p95_index = min(len(sorted_times) - 1, int(len(sorted_times) * 0.95))
        summary[category] = {
            "count": len(times),
            "errors": errors_by_category[category],
            "mean": mean(times),
            "median": median(times),
            "min": min(times),
            "max": max(times),
            "p95": sorted_times[p95_index],
        }
    return summary


def save_results_json(
    results: list[TrialResult], path: Path, provider: str, model: str
) -> None:
    payload = {
        "provider": provider,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summarize(results),
        "trials": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_results(results: list[TrialResult], output_path: Path) -> None:
    surface = "#fcfcfb"
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"
    ink_muted = "#898781"
    grid_color = "#e1e0d9"
    axis_color = "#c3c2b7"

    fig, (ax_time, ax_len) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(surface)

    for ax in (ax_time, ax_len):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid_color, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(axis_color)
        ax.spines["bottom"].set_color(axis_color)
        ax.tick_params(colors=ink_muted, labelsize=9)

    for category in PROMPT_BUILDERS:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue
        color = CATEGORY_COLORS[category]
        label = CATEGORY_LABELS[category]

        ax_time.plot(
            [r.index for r in cat_results],
            [r.elapsed_seconds for r in cat_results],
            marker="o", markersize=4, linewidth=1.2,
            color=color, label=label, zorder=3,
        )
        ax_len.scatter(
            [r.prompt_chars for r in cat_results],
            [r.elapsed_seconds for r in cat_results],
            s=24, color=color, label=label, zorder=3, edgecolors="none",
        )

    ax_time.set_yscale("log")
    ax_time.set_xlabel("Request #", color=ink_secondary, fontsize=10)
    ax_time.set_ylabel("Answer time (s, log scale)",
                       color=ink_secondary, fontsize=10)
    ax_time.set_title("Answer time per request",
                      color=ink_primary, fontsize=12, loc="left")
    ax_time.legend(frameon=False, fontsize=9, labelcolor=ink_secondary)

    ax_len.set_xscale("log")
    ax_len.set_yscale("log")
    ax_len.set_xlabel("Prompt length (chars, log scale)",
                      color=ink_secondary, fontsize=10)
    ax_len.set_ylabel("Answer time (s, log scale)",
                      color=ink_secondary, fontsize=10)
    ax_len.set_title("Answer time vs. prompt length",
                     color=ink_primary, fontsize=12, loc="left")
    ax_len.legend(frameon=False, fontsize=9, labelcolor=ink_secondary)

    fig.suptitle("LLM endpoint latency benchmark",
                 color=ink_primary, fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=surface, bbox_inches="tight")
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark answer-time latency of the configured LLM endpoint."
    )
    parser.add_argument(
        "--num-requests", type=int, default=DEFAULT_NUM_REQUESTS,
        help=f"Requests per prompt category (default: {DEFAULT_NUM_REQUESTS}).",
    )
    parser.add_argument(
        "--provider", choices=sorted(config.LLM_PROVIDER_DICTIONARY), default=config.LLM_PROVIDER,
        help="LLM client provider to benchmark (default: the agents' configured provider).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Delay in seconds between consecutive requests (default: 0).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results",
        help="Directory to write the JSON log and PNG plot to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_dotenv()

    client_cls = config.LLM_PROVIDER_DICTIONARY[args.provider]
    client: AbstractLLMClient = client_cls()

    results = asyncio.run(run_benchmark(client, args.num_requests, args.delay))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"bench_{args.provider}_{timestamp}.json"
    plot_path = args.output_dir / f"bench_{args.provider}_{timestamp}.png"

    save_results_json(results, json_path, args.provider, client.model())
    print(f"\nSaved raw results to {json_path}")

    for category, stats in summarize(results).items():
        print(
            f"{CATEGORY_LABELS[category]:22s} "
            f"mean={stats['mean']:6.2f}s median={stats['median']:6.2f}s "
            f"p95={stats['p95']:6.2f}s max={stats['max']:6.2f}s errors={stats['errors']}"
        )

    plot_results(results, plot_path)
    print(f"Saved plot to {plot_path}")

    if client.rate_limited():
        print("Warning: the client hit rate limiting during the benchmark; see retry_log for details.")


if __name__ == "__main__":
    main()
