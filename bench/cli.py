"""CLI entry point for latent-consistency-bench."""

from __future__ import annotations

import argparse
import hashlib
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

        digest = hashlib.sha256(("baseline:" + prompt).encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))
        return Image.fromarray(rng.randint(0, 256, (512, 512, 3), dtype=np.uint8))

    def mock_qbp(prompt: str, **kwargs):
        import numpy as np
        from PIL import Image

        digest = hashlib.sha256(("synthetic-qbp:" + prompt).encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))
        return Image.fromarray(rng.randint(0, 256, (512, 512, 3), dtype=np.uint8))

    bench = ConsistencyBenchmark(mock_baseline, mock_qbp)
    if args.characters < 1 or args.environments < 1:
        raise SystemExit("--characters and --environments must both be >= 1")
    print("DEMO: deterministic synthetic images, not real model/QBP results.")
    print(f"Running character consistency ({args.characters} subjects)...")
    char_result = bench.run_character_consistency(n=args.characters)
    print(f"Running temporal consistency ({args.environments} sequences)...")
    temp_result = bench.run_temporal_consistency(n=args.environments)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    combined = {
        "kind": "synthetic-demo",
        "character": char_result.to_dict(),
        "temporal": temp_result.to_dict(),
    }
    output.write_text(json.dumps(combined, indent=2))
    print(f"Results saved to {output}")

    fig = generate_chart([char_result, temp_result], title="LCB Benchmark Results")
    chart_path = output.with_suffix(".png")
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")


def _load_result(path: str, section: str | None = None) -> BenchmarkResult:
    """Read a benchmark result or explicitly choose a section of a run file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Results file must be a JSON object")
    if "scores" not in data:
        sections = [name for name in ("character", "temporal") if name in data]
        if not sections:
            raise ValueError("No benchmark scores found")
        if section is None:
            raise ValueError("Combined results contain multiple sections; select --section character or temporal")
        if section not in sections:
            raise ValueError(f"Section '{section}' not available; choose one of {sections}")
        data = data[section]
    elif section is not None:
        raise ValueError("--section is only valid for combined run files")
    if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
        raise ValueError("Selected results have no valid scores")
    return BenchmarkResult(
        name=data.get("name", "result"),
        scores=data["scores"],
        metadata=data.get("metadata", {}),
        num_samples=data.get("num_samples", 0),
    )


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare the selected sections of two result files."""
    result_a = _load_result(args.baseline, args.section)
    result_b = _load_result(args.qbp, args.section)
    report = compare_results(result_a, result_b)
    print(report.to_markdown())


def cmd_report(args: argparse.Namespace) -> None:
    """Generate report from a single result or selected combined section."""
    result = _load_result(args.results, args.section)

    fmt = args.format
    if fmt == "html":
        out = result.to_html()
        ext = ".html"
    elif fmt == "csv":
        source_path = Path(args.results)
        csv_path = source_path.with_name(source_path.stem + (("-" + args.section) if args.section else "") + ".csv")
        export_csv([result], csv_path)
        print(f"CSV saved to {csv_path}")
        return
    else:
        out = result.to_markdown()
        ext = ".md"

    source_path = Path(args.results)
    output_path = source_path.with_name(source_path.stem + (("-" + args.section) if args.section else "") + ext)
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
    compare_parser.add_argument("--section", choices=["character", "temporal"], help="Section for combined run results")

    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument("results", help="Results JSON file")
    report_parser.add_argument("--section", choices=["character", "temporal"], help="Section for combined run results")
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
