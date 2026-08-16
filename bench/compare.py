"""Comparison and visualization tools for benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bench.report import BenchmarkResult, ComparisonReport


def compare_results(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
) -> ComparisonReport:
    """Compare two benchmark results and produce a report.

    Args:
        result_a: First result (typically baseline).
        result_b: Second result (typically QBP-enhanced).

    Returns:
        ComparisonReport with deltas and winner.
    """
    return ComparisonReport(result_a=result_a, result_b=result_b)


def generate_chart(
    results: Sequence[BenchmarkResult],
    *,
    title: str = "Benchmark Comparison",
    figsize: tuple[int, int] = (10, 6),
) -> Any:
    """Generate a matplotlib grouped bar chart comparing results.

    Args:
        results: Two or more BenchmarkResult objects to compare.
        title: Chart title.
        figsize: Figure size as (width, height) in inches.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("At least one result required")

    all_dims: list[str] = sorted(
        {d for r in results for d in r.scores}
    )
    n_dims = len(all_dims)
    n_results = len(results)
    bar_width = 0.8 / n_results

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_dims)

    colors = ["#4a9eff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8"]
    for i, result in enumerate(results):
        vals = [result.scores.get(d, 0.0) for d in all_dims]
        offset = (i - n_results / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=result.name,
            color=colors[i % len(colors)],
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(all_dims, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def export_csv(results: Sequence[BenchmarkResult], path: str | Path) -> Path:
    """Export benchmark results to CSV.

    Args:
        results: Results to export.
        path: Output file path.

    Returns:
        Path to the written CSV file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_dims: list[str] = sorted({d for r in results for d in r.scores})
    headers = ["name", "overall", "num_samples"] + all_dims

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            row = [r.name, f"{r.overall:.4f}", r.num_samples]
            row += [f"{r.scores.get(d, 0.0):.4f}" for d in all_dims]
            writer.writerow(row)

    return path
