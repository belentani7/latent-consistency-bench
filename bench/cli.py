"""CLI entry point for latent-consistency-bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.compare import compare_results, export_csv, generate_chart
from bench.report import BenchmarkResult


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmark suite."""
    from bench.benchmark import ConsistencyBenchmark

    def mock_baseline(prompt: str, **kwargs):
        import numpy as np
        from PIL import Image

        rng = np.random.RandomState(hash(prompt) % (2**31))
        return Image.fromarray(rng.randint(0, 256, (512, 512, 3), dtype=np.uint8))

    def mock_qbp(prompt: str, **kwargs):
        import numpy as np
        from PIL import Image

        rng = np.random.RandomState(hash(prompt) % (2**31) + 1)
        return Image.fromarray(rng.randint(0, 256, (512, 512, 3), dtype=np.uint8))

    bench = ConsistencyBenchmark(mock_baseline, mock_qbp)
    print(f"Running character consistency ({args.characters} subjects)...")
    char_result = bench.run_character_consistency(n=args.characters)
    print(f"Running temporal consistency ({args.environments} sequences)...")
    temp_result = bench.run_temporal_consistency(n=args.environments)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    combined = {
        "character": char_result.to_dict(),
        "temporal": temp_result.to_dict(),
    }
    output.write_text(json.dumps(combined, indent=2))
    print(f"Results saved to {output}")

    fig = generate_chart([char_result, temp_result], title="LCB Benchmark Results")
    chart_path = output.with_suffix(".png")
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two result files."""
    with open(args.baseline) as f:
        data_a = json.load(f)
    with open(args.qbp) as f:
        data_b = json.load(f)

    result_a = BenchmarkResult(
        name=data_a.get("name", "baseline"),
        scores=data_a.get("scores", {}),
        metadata=data_a.get("metadata", {}),
        num_samples=data_a.get("num_samples", 0),
    )
    result_b = BenchmarkResult(
        name=data_b.get("name", "qbp"),
        scores=data_b.get("scores", {}),
        metadata=data_b.get("metadata", {}),
        num_samples=data_b.get("num_samples", 0),
    )

    report = compare_results(result_a, result_b)
    print(report.to_markdown())


def cmd_report(args: argparse.Namespace) -> None:
    """Generate report from results file."""
    with open(args.results) as f:
        data = json.load(f)

    result = BenchmarkResult(
        name=data.get("name", "results"),
        scores=data.get("scores", {}),
        metadata=data.get("metadata", {}),
        num_samples=data.get("num_samples", 0),
    )

    fmt = args.format
    if fmt == "html":
        out = result.to_html()
        ext = ".html"
    elif fmt == "csv":
        csv_path = Path(args.results).with_suffix(".csv")
        export_csv([result], csv_path)
        print(f"CSV saved to {csv_path}")
        return
    else:
        out = result.to_markdown()
        ext = ".md"

    output_path = Path(args.results).with_suffix(ext)
    output_path.write_text(out)
    print(f"Report saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lcb",
        description="Latent Consistency Bench - AI visual consistency benchmarking",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run benchmark suite")
    run_parser.add_argument("--characters", type=int, default=20, help="Number of character subjects")
    run_parser.add_argument("--environments", type=int, default=10, help="Number of environment sequences")
    run_parser.add_argument("--output", default="results.json", help="Output file path")

    compare_parser = subparsers.add_parser("compare", help="Compare two result files")
    compare_parser.add_argument("baseline", help="Baseline results JSON")
    compare_parser.add_argument("qbp", help="QBP results JSON")

    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument("results", help="Results JSON file")
    report_parser.add_argument("--format", choices=["html", "csv", "md"], default="html", help="Output format")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
