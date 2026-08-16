"""Report dataclasses and formatting for benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    num_samples: int = 0

    @property
    def overall(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall"] = self.overall
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark Result: {self.name}",
            "",
            f"**Samples:** {self.num_samples}",
            f"**Overall Score:** {self.overall:.4f}",
            "",
            "## Dimension Scores",
            "",
            "| Dimension | Score |",
            "|-----------|-------|",
        ]
        for dim, score in sorted(self.scores.items()):
            lines.append(f"| {dim} | {score:.4f} |")
        if self.metadata:
            lines.extend(["", "## Metadata", ""])
            for k, v in self.metadata.items():
                lines.append(f"- **{k}:** {v}")
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = ""
        for dim, score in sorted(self.scores.items()):
            bar_width = int(score * 100)
            rows += (
                f'<tr><td>{dim}</td>'
                f'<td>{score:.4f}</td>'
                f'<td><div class="bar" style="width:{bar_width}%"></div></td></tr>\n'
            )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark: {self.name}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #ddd; }}
.bar {{ height: 16px; background: #4a9eff; border-radius: 3px; }}
h1 {{ color: #1a1a2e; }}
.overall {{ font-size: 1.4rem; font-weight: bold; color: #16213e; }}
</style>
</head>
<body>
<h1>Benchmark Result: {self.name}</h1>
<p class="overall">Overall: {self.overall:.4f} ({self.num_samples} samples)</p>
<table>
<tr><th>Dimension</th><th>Score</th><th>Visual</th></tr>
{rows}
</table>
</body>
</html>"""


@dataclass
class ComparisonReport:
    """Comparison of two benchmark results."""

    result_a: BenchmarkResult
    result_b: BenchmarkResult
    deltas: dict[str, float] = field(default_factory=dict)
    winner: str = ""

    def __post_init__(self) -> None:
        if not self.deltas:
            all_dims = set(self.result_a.scores) | set(self.result_b.scores)
            self.deltas = {
                d: self.result_b.scores.get(d, 0.0) - self.result_a.scores.get(d, 0.0)
                for d in sorted(all_dims)
            }
        if not self.winner:
            if self.result_a.overall > self.result_b.overall:
                self.winner = self.result_a.name
            elif self.result_b.overall > self.result_a.overall:
                self.winner = self.result_b.name
            else:
                self.winner = "tie"

    def to_markdown(self) -> str:
        lines = [
            f"# Comparison: {self.result_a.name} vs {self.result_b.name}",
            "",
            f"**Winner:** {self.winner}",
            "",
            "## Scores",
            "",
            "| Dimension | A | B | Delta |",
            "|-----------|---|---|-------|",
        ]
        for dim in sorted(self.deltas):
            a = self.result_a.scores.get(dim, 0.0)
            b = self.result_b.scores.get(dim, 0.0)
            d = self.deltas[dim]
            sign = "+" if d > 0 else ""
            lines.append(f"| {dim} | {a:.4f} | {b:.4f} | {sign}{d:.4f} |")
        lines.extend([
            "",
            f"**Overall A:** {self.result_a.overall:.4f}",
            f"**Overall B:** {self.result_b.overall:.4f}",
        ])
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = ""
        for dim in sorted(self.deltas):
            a = self.result_a.scores.get(dim, 0.0)
            b = self.result_b.scores.get(dim, 0.0)
            d = self.deltas[dim]
            cls = "pos" if d > 0 else ("neg" if d < 0 else "")
            sign = "+" if d > 0 else ""
            rows += f'<tr><td>{dim}</td><td>{a:.4f}</td><td>{b:.4f}</td><td class="{cls}">{sign}{d:.4f}</td></tr>\n'
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Comparison: {self.result_a.name} vs {self.result_b.name}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #ddd; }}
.pos {{ color: #2e7d32; font-weight: bold; }}
.neg {{ color: #c62828; font-weight: bold; }}
.winner {{ font-size: 1.4rem; color: #16213e; font-weight: bold; }}
</style>
</head>
<body>
<h1>Comparison: {self.result_a.name} vs {self.result_b.name}</h1>
<p class="winner">Winner: {self.winner}</p>
<table>
<tr><th>Dimension</th><th>{self.result_a.name}</th><th>{self.result_b.name}</th><th>Delta</th></tr>
{rows}
</table>
</body>
</html>"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_a": self.result_a.to_dict(),
            "result_b": self.result_b.to_dict(),
            "deltas": self.deltas,
            "winner": self.winner,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
