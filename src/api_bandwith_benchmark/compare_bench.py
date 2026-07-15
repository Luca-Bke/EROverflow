"""Compare two api_bench.py result JSON files as grouped box plots.

Run directly:
    `python src/api_bandwith_benchmark/compare_bench.py <file_a> <file_b>
    [options]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Kept in sync with api_bench.py's category scheme so labels read consistently
# with single-run plots.
CATEGORY_LABELS = {
    "single_prompt": "Single prompt",
    "multi_prompt": "Multi prompt",
    "long_context_prompt": "Long context prompt",
}
CATEGORY_ORDER = list(CATEGORY_LABELS)

SOURCE_COLORS = ["#2a78d6", "#eda100"]  # blue, yellow — one per file


def load_trials(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, payload["trials"]


def default_label(payload: dict, path: Path) -> str:
    return f"{payload.get('provider', '?')}/{payload.get('model', path.stem)}"


def print_summary_comparison(
    payload_a: dict, label_a: str, payload_b: dict, label_b: str
) -> None:
    categories = dict.fromkeys(
        list(payload_a.get("summary", {})) + list(payload_b.get("summary", {}))
    )
    print(f"\n{'Category':22s} {'metric':8s} {label_a:>15s} {label_b:>15s}")
    for category in categories:
        stats_a = payload_a.get("summary", {}).get(category)
        stats_b = payload_b.get("summary", {}).get(category)
        label = CATEGORY_LABELS.get(category, category)
        for metric in ("mean", "median", "p95", "max"):
            val_a = f"{stats_a[metric]:.2f}s" if stats_a else "-"
            val_b = f"{stats_b[metric]:.2f}s" if stats_b else "-"
            print(f"{label:22s} {metric:8s} {val_a:>15s} {val_b:>15s}")


def ordered_categories(
    trials_a: list[dict], trials_b: list[dict]
) -> list[str]:
    present = {t["category"] for t in trials_a}
    present |= {t["category"] for t in trials_b}
    ordered = [c for c in CATEGORY_ORDER if c in present]
    # Unknown categories still get shown, appended at the end.
    ordered += sorted(present - set(ordered))
    return ordered


def plot_comparison(
    trials_a: list[dict], label_a: str,
    trials_b: list[dict], label_b: str,
    output_path: Path,
) -> None:
    surface = "#fcfcfb"
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"
    ink_muted = "#898781"
    grid_color = "#e1e0d9"
    axis_color = "#c3c2b7"

    categories = ordered_categories(trials_a, trials_b)

    def times_for(trials: list[dict], category: str) -> list[float]:
        return [
            t["elapsed_seconds"] for t in trials if t["category"] == category
        ]

    data_a = [times_for(trials_a, c) for c in categories]
    data_b = [times_for(trials_b, c) for c in categories]

    group_width = 1.6
    box_width = 0.55
    positions_a = [i * group_width for i in range(len(categories))]
    positions_b = [p + box_width + 0.15 for p in positions_a]
    xtick_positions = [
        (pa + pb) / 2 for pa, pb in zip(positions_a, positions_b)
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.grid(True, axis="y", color=grid_color, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_color(axis_color)
    ax.tick_params(colors=ink_muted, labelsize=9)

    flier_style = dict(
        marker="o", markersize=4, alpha=0.55, markeredgecolor="none"
    )

    def style_boxplot(bp, color):
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set_color(color)
        for line in bp["medians"]:
            line.set_color(ink_primary)
            line.set_linewidth(1.4)
        for flier in bp["fliers"]:
            flier.set_markerfacecolor(color)

    bp_a = ax.boxplot(
        data_a, positions=positions_a, widths=box_width,
        patch_artist=True, showfliers=True,
        flierprops=flier_style, zorder=3,
    )
    bp_b = ax.boxplot(
        data_b, positions=positions_b, widths=box_width,
        patch_artist=True, showfliers=True,
        flierprops=flier_style, zorder=3,
    )
    style_boxplot(bp_a, SOURCE_COLORS[0])
    style_boxplot(bp_b, SOURCE_COLORS[1])

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(
        [CATEGORY_LABELS.get(c, c) for c in categories],
        fontsize=10, color=ink_secondary,
    )
    ax.set_yscale("log")
    ax.set_ylabel(
        "Answer time (s, log scale)", color=ink_secondary, fontsize=10
    )

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=SOURCE_COLORS[0], alpha=0.7,
            label=label_a,
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=SOURCE_COLORS[1], alpha=0.7,
            label=label_b,
        ),
    ]
    ax.legend(
        handles=legend_handles, frameon=False, fontsize=9,
        labelcolor=ink_secondary,
    )

    fig.suptitle(f"LLM endpoint latency comparison: {label_a} vs {label_b}",
                 color=ink_primary, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=surface, bbox_inches="tight")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two api_bench.py JSON result files as grouped box plots."
        )
    )
    parser.add_argument("file_a", type=Path, help="First bench_*.json file.")
    parser.add_argument("file_b", type=Path, help="Second bench_*.json file.")
    parser.add_argument(
        "--label-a", type=str, default=None,
        help=(
            "Legend label for the first file "
            "(default: provider/model from the JSON)."
        ),
    )
    parser.add_argument(
        "--label-b", type=str, default=None,
        help=(
            "Legend label for the second file "
            "(default: provider/model from the JSON)."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Path to save the comparison PNG "
            "(default: results/compare_<a>_vs_<b>.png)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload_a, trials_a = load_trials(args.file_a)
    payload_b, trials_b = load_trials(args.file_b)

    label_a = args.label_a or default_label(payload_a, args.file_a)
    label_b = args.label_b or default_label(payload_b, args.file_b)

    output_path = args.output or (
        Path(__file__).resolve().parent / "results"
        / f"compare_{args.file_a.stem}_vs_{args.file_b.stem}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_summary_comparison(payload_a, label_a, payload_b, label_b)
    plot_comparison(trials_a, label_a, trials_b, label_b, output_path)
    print(f"\nSaved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
