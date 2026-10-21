"""CLI for reproducible *simulated* consistency tests and real report exports.

The built-in generators create deterministic noise images. They do not call
ComfyUI, a diffusion pipeline, CLIP or real model providers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from html import escape
from pathlib import Path

from bench.compare import compare_results, export_csv, generate_chart
from bench.report import BenchmarkResult

SECTIONS = ("character", "temporal")


def _parse_result(raw: object, *, name: str) -> BenchmarkResult:
    if not isinstance(raw, dict) or not isinstance(raw.get("scores"), dict):
        raise ValueError(f"{name}: expected a result object with scores")
    scores = {}
    for dimension, value in raw["scores"].items():
        if (not isinstance(dimension, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1):
            raise ValueError(f"{name}: invalid score {dimension!r}")
        scores[dimension] = float(value)
    count = raw.get("num_samples", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{name}: invalid sample count")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{name}: invalid metadata")
    return BenchmarkResult(
        name=str(raw.get("name") or name),
        scores=scores, metadata=metadata, num_samples=count,
    )


def load_results(path: str | Path) -> dict[str, BenchmarkResult]:
    """Open native suite JSON from 'lcb run' or legacy single result JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("result file must be a JSON object")
    if any(section in data for section in SECTIONS):
        sections = {
            key: _parse_result(data[key], name=key)
            for key in SECTIONS if key in data
        }
        if not sections:
            raise ValueError("empty benchmark suite")
        return sections
    return {"result": _parse_result(data, name="result")}


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a deterministic offline simulation; never present as a real model test."""
    from PIL import Image
    import numpy as np
    from bench.benchmark import ConsistencyBenchmark

    if args.characters < 1 or args.environments < 1:
        raise ValueError("--characters and --environments must be >= 1")

    def sample_image(prompt: str, variant: int):
        payload = f"{args.seed}:{variant}:{prompt}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        rng = np.random.RandomState(seed)
        return Image.fromarray(rng.randint(0, 256, (128, 128, 3), dtype=np.uint8))

    bench = ConsistencyBenchmark(
        lambda prompt, **kwargs: sample_image(prompt, 0),
        lambda prompt, **kwargs: sample_image(prompt, 1),
        seed=args.seed,
    )
    print("MODE: synthetic demonstration, NOT a benchmark of real AI generators")
    print(f"Character samples: {args.characters}; temporal sequences: {args.environments}")
    char_result = bench.run_character_consistency(n=args.characters)
    temp_result = bench.run_temporal_consistency(n=args.environments)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "format_version": 1,
        "simulation": True,
        "seed": args.seed,
        "character": char_result.to_dict(),
        "temporal": temp_result.to_dict(),
    }
    output.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON report: {output}")

    fig = generate_chart([char_result, temp_result], title="LCB · Synthetic Demonstration")
    chart_path = output.with_suffix(".png")
    try:
        fig.savefig(chart_path, dpi=150)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"Chart: {chart_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare either two single results or two matching suite sections."""
    baseline = load_results(args.baseline)
    enhanced = load_results(args.qbp)
    common = sorted(set(baseline) & set(enhanced))
    if not common:
        raise ValueError("No matching result types in the two JSON files")
    for key in common:
        print(f"\n## Section: {key}\n")
        print(compare_results(baseline[key], enhanced[key]).to_markdown())
    if set(baseline) != set(enhanced):
        print("Warning: unmatched suite sections were omitted", file=sys.stderr)


def _suite_html(results: dict[str, BenchmarkResult]) -> str:
    body = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>LCB report</title>",
        "<style>body{background:#080f17;color:#edf7fc;font:16px system-ui;max-width:980px;margin:auto;padding:35px}",
        "h1,h2{letter-spacing:-.04em}table{width:100%;border-collapse:collapse;margin:20px 0 48px}",
        "td,th{text-align:left;border-bottom:1px solid #ffffff25;padding:12px}th{color:#a3e9ff}",
        "p{color:#c2d0d6}a{color:#b5efff}</style></head><body>",
        "<h1>LATENT CONSISTENCY BENCH / REPORT</h1>",
        "<p>Scores from the supplied JSON. Synthetic results are not evidence of real model performance.</p>",
    ]
    for label, result in results.items():
        body.append(f"<section><h2>{escape(label.title())} — {escape(result.name)}</h2>")
        body.append(f"<p>Samples: {result.num_samples} · Overall (simple mean): {result.overall:.4f}</p>")
        body.append("<table><thead><tr><th>Metric</th><th>Score</th></tr></thead><tbody>")
        for dimension, score in sorted(result.scores.items()):
            body.append(f"<tr><td>{escape(dimension)}</td><td>{score:.4f}</td></tr>")
        body.append("</tbody></table></section>")
    return "\n".join(body + ["</body></html>"])


def cmd_report(args: argparse.Namespace) -> None:
    """Export *all* sections of the JSON produced by 'run'."""
    results = load_results(args.results)
    fmt = args.format
    base = Path(args.results)
    if fmt == "csv":
        csv_path = base.with_suffix(".csv")
        export_csv(list(results.values()), csv_path)
        print(f"CSV report: {csv_path}")
        return
    if fmt == "html":
        rendered = _suite_html(results)
        extension = ".html"
    else:
        rendered = "\n\n---\n\n".join(
            f"## {name.title()}\n\n{result.to_markdown()}"
            for name, result in results.items()
        )
        extension = ".md"
    path = base.with_suffix(extension)
    path.write_text(rendered, encoding="utf-8")
    print(f"Report: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lcb", description="Latent Consistency Bench — offline simulation and analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run reproducible synthetic example")
    run_parser.add_argument("--characters", type=int, default=20)
    run_parser.add_argument("--environments", type=int, default=10)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--output", default="results.json")

    compare_parser = subparsers.add_parser("compare", help="Compare JSON exports")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("qbp")

    report_parser = subparsers.add_parser("report", help="Export entire result file")
    report_parser.add_argument("results")
    report_parser.add_argument("--format", choices=["html", "csv", "md"], default="html")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            cmd_run(args)
        elif args.command == "compare":
            cmd_compare(args)
        else:
            cmd_report(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"lcb: {exc}\n")


if __name__ == "__main__":
    main()
