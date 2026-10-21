"""Combined result report and output escaping regressions."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bench.cli import _load_result, cmd_report
from bench.report import BenchmarkResult, ComparisonReport
from bench.benchmark import ConsistencyBenchmark


def test_combined_results_require_explicit_section(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "kind": "synthetic-demo",
        "character": {"name": "character", "scores": {"identity": 0.7}, "num_samples": 2},
        "temporal": {"name": "temporal", "scores": {"temporal": 0.8}, "num_samples": 1}
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="--section"):
        _load_result(str(path))
    result = _load_result(str(path), "character")
    assert result.num_samples == 2
    cmd_report(Namespace(results=str(path), section="character", format="md"))
    assert (tmp_path / "results-character.md").exists()


def test_html_escapes_names_and_dimension_labels():
    result = BenchmarkResult(
        name="<script>alert(1)</script>",
        scores={"<img src=x onerror=alert(1)>": 0.7},
        num_samples=1,
    )
    html = result.to_html()
    assert "<script>" not in html
    assert "<img " not in html
    assert "&lt;script&gt;" in html
    comparison = ComparisonReport(result, BenchmarkResult(name="safe", scores=result.scores))
    assert "<script>" not in comparison.to_html()


def test_benchmark_init_does_not_change_global_random_state():
    numpy_before = np.random.get_state()
    generated = lambda prompt: Image.new("RGB", (8, 8), "black")
    ConsistencyBenchmark(generated, generated, seed=314)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
