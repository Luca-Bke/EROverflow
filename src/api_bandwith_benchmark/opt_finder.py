"""Binary search for the maximum prompt context length the LLM endpoint
accepts before it starts erroring.

api_bench.py's `long_context_prompt` category (~40k chars, built via
`build_long_context_prompt`) reliably triggered a 500 "Model returned empty
generation output" error after retries were exhausted, while `multi_prompt`
(~2.6k chars) succeeded every time. This script reuses the same
`build_long_context_prompt` builder from api_bench.py — which is
parameterized by `num_log_messages` and therefore already lets us grow the
prompt in controlled steps — and binary-searches over that parameter to find
where the endpoint breaks, without touching api_bench.py itself.

Run directly: `python src/api_bandwith_benchmark/opt_finder.py [options]`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Make the `agents` and `api_bandwith_benchmark` packages importable when
# this file is run directly.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agents.configuration import config  # noqa: E402
from agents.llm_clients.abstract_llm_client import AbstractLLMClient  # noqa: E402
from api_bandwith_benchmark.api_bench import build_long_context_prompt  # noqa: E402

DEFAULT_START_MESSAGES = 1
DEFAULT_MAX_MESSAGES = 64
DEFAULT_REPEATS = 1


# ── Environment setup ─────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """Populate os.environ from the repo-root .env file, if present."""
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


def _make_client(provider: str) -> AbstractLLMClient:
    client_cls = config.LLM_PROVIDER_DICTIONARY[provider]
    try:
        # A probe only needs a fast yes/no, not the client's own retry policy
        # (which would hide failures behind minutes of exponential backoff).
        return client_cls(backoff_enabled=False)
    except TypeError:
        return client_cls()


# ── Probing ────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    step: int
    num_log_messages: int
    prompt_chars: int
    success: bool
    elapsed_seconds: float
    repeats: int
    successes: int
    error: str | None = None


async def probe_once(client: AbstractLLMClient, num_log_messages: int) -> tuple[int, bool, float, str | None]:
    messages = build_long_context_prompt(num_log_messages)
    prompt_chars = sum(len(str(m.content)) for m in messages)
    start = time.perf_counter()
    try:
        await client.invoke_async(messages)
        return prompt_chars, True, time.perf_counter() - start, None
    except Exception as e:  # noqa: BLE001 - any failure counts as a broken probe
        return prompt_chars, False, time.perf_counter() - start, str(e)[:300]


async def probe(client: AbstractLLMClient, num_log_messages: int, step: int, repeats: int) -> Probe:
    """Probe one context size, repeating and majority-voting for robustness."""
    successes = 0
    total_elapsed = 0.0
    prompt_chars = 0
    last_error: str | None = None
    for _ in range(repeats):
        prompt_chars, ok, elapsed, error = await probe_once(client, num_log_messages)
        total_elapsed += elapsed
        if ok:
            successes += 1
        else:
            last_error = error
    success = successes * 2 > repeats
    result = Probe(
        step, num_log_messages, prompt_chars, success, total_elapsed,
        repeats, successes, None if success else last_error,
    )
    status = "ok" if success else "FAIL"
    print(
        f"  probe {step}: n={num_log_messages:3d} log messages "
        f"({prompt_chars:6d} chars) -> {status} "
        f"({successes}/{repeats} succeeded, {total_elapsed:.1f}s)"
        + (f" - {last_error}" if not success and last_error else "")
    )
    return result


async def find_context_limit(
    client: AbstractLLMClient, start_messages: int, max_messages: int, repeats: int
) -> tuple[list[Probe], int | None, int | None]:
    """Binary-search `num_log_messages` for the largest value that still succeeds.

    Returns (all probes, max good num_log_messages or None, min bad num_log_messages
    or None if no failure was observed up to max_messages).
    """
    probes: list[Probe] = []
    step = 1

    baseline = await probe(client, 0, step, repeats)
    probes.append(baseline)
    if not baseline.success:
        # Even the shortest long-context-style prompt fails; there is nothing
        # to bisect over context length here.
        return probes, None, 0

    low = 0
    high: int | None = None
    n = max(1, start_messages)

    # Phase 1: exponential growth to bracket a failing length.
    while n <= max_messages:
        step += 1
        result = await probe(client, n, step, repeats)
        probes.append(result)
        if result.success:
            low = n
            n *= 2
        else:
            high = n
            break

    if high is None:
        return probes, low, None

    # Phase 2: binary search between the last good (low) and first bad (high).
    while high - low > 1:
        mid = (low + high) // 2
        step += 1
        result = await probe(client, mid, step, repeats)
        probes.append(result)
        if result.success:
            low = mid
        else:
            high = mid

    return probes, low, high


# ── Reporting ──────────────────────────────────────────────────────────────────

def save_probes_json(
    probes: list[Probe], path: Path, provider: str, model: str,
    max_good_n: int | None, min_bad_n: int | None,
) -> None:
    good_chars = next((p.prompt_chars for p in probes if p.num_log_messages == max_good_n and p.success), None)
    bad_chars = next((p.prompt_chars for p in probes if p.num_log_messages == min_bad_n and not p.success), None)
    payload = {
        "provider": provider,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_good_num_log_messages": max_good_n,
        "max_good_prompt_chars": good_chars,
        "min_bad_num_log_messages": min_bad_n,
        "min_bad_prompt_chars": bad_chars,
        "probes": [asdict(p) for p in probes],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_by_length(probes: list[Probe]) -> list[dict]:
    """Per num_log_messages: how many individual calls were made, and the
    success/failure split among them (a probe's majority vote can hide a
    near-miss, e.g. 2/3 succeeding)."""
    lengths = []
    for p in sorted(probes, key=lambda p: p.num_log_messages):
        lengths.append({
            "num_log_messages": p.num_log_messages,
            "prompt_chars": p.prompt_chars,
            "repeats": p.repeats,
            "successes": p.successes,
            "failures": p.repeats - p.successes,
            "success_rate": p.successes / p.repeats,
            "majority_success": p.success,
            "elapsed_seconds_total": p.elapsed_seconds,
            "elapsed_seconds_mean": p.elapsed_seconds / p.repeats,
            "error": p.error,
        })
    return lengths


def save_length_summary_json(
    probes: list[Probe], path: Path, provider: str, model: str,
) -> None:
    payload = {
        "provider": provider,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lengths": summarize_by_length(probes),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary-search the maximum prompt context length the LLM endpoint accepts."
    )
    parser.add_argument(
        "--provider", choices=sorted(config.LLM_PROVIDER_DICTIONARY), default=config.LLM_PROVIDER,
        help="LLM client provider to probe (default: the agents' configured provider).",
    )
    parser.add_argument(
        "--start-messages", type=int, default=DEFAULT_START_MESSAGES,
        help=f"Starting num_log_messages for the exponential growth phase (default: {DEFAULT_START_MESSAGES}).",
    )
    parser.add_argument(
        "--max-messages", type=int, default=DEFAULT_MAX_MESSAGES,
        help=f"Upper cap on num_log_messages to try before giving up (default: {DEFAULT_MAX_MESSAGES}).",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"Repeat each probe this many times and majority-vote the result (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results",
        help="Directory to write the probe JSON log to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_dotenv()

    client = _make_client(args.provider)

    print(f"Searching max context length for provider={args.provider} model={client.model()}")
    probes, max_good_n, min_bad_n = asyncio.run(
        find_context_limit(client, args.start_messages, args.max_messages, args.repeats)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"context_limit_{args.provider}_{timestamp}.json"
    save_probes_json(probes, json_path, args.provider, client.model(), max_good_n, min_bad_n)
    print(f"\nSaved probe log to {json_path}")

    summary_path = args.output_dir / f"context_limit_summary_{args.provider}_{timestamp}.json"
    save_length_summary_json(probes, summary_path, args.provider, client.model())
    print(f"Saved per-length summary to {summary_path}")

    if max_good_n is None:
        print("Even the shortest long-context-style prompt failed — this looks like an "
              "endpoint/auth issue rather than a context-length limit.")
    elif min_bad_n is None:
        print(f"No failure observed up to num_log_messages={args.max_messages}; "
              "raise --max-messages to keep searching.")
    else:
        good_chars = next(p.prompt_chars for p in probes if p.num_log_messages == max_good_n and p.success)
        bad_chars = next(p.prompt_chars for p in probes if p.num_log_messages == min_bad_n and not p.success)
        print(
            f"Maximum safe context length: {good_chars} chars ({max_good_n} log-history messages).\n"
            f"First failing length observed: {bad_chars} chars ({min_bad_n} log-history messages)."
        )


if __name__ == "__main__":
    main()
