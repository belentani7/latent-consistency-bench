"""Round-trip tests for the exact CLI payload produced by 'lcb run'."""
from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from bench.cli import _suite_html, cmd_compare, cmd_report, cmd_run, load_results
from bench.report import BenchmarkResult


def test_can_read_legacy_single_result(tmp_path):
    file = tmp_path / "single.json"
    file.write_text(json.dumps({"name": "old", "scores": {"identity": 0.7}, "num_samples": 1}))
    result = load_results(file)
    assert list(result) == ["result"]
    assert result["result"].scores["identity"] == .7


def test_run_produces_reproducible_composite_json_and_png(tmp_path):
    a = tmp_path / "one.json"
    b = tmp_path / "two.json"
    cmd_run(Namespace(characters=1, environments=1, materials=1, output=str(a), seed=42))
    cmd_run(Namespace(characters=1, environments=1, materials=1, output=str(b), seed=42))
    assert json.loads(a.read_text()) == json.loads(b.read_text())
    assert a.with_suffix(".png").stat().st_size > 100
    assert b.with_suffix(".png").stat().st_size > 100
    raw = json.loads(a.read_text())
    assert raw["simulation"] is True
    assert list(load_results(a)) == ["character", "temporal", "pbr"]


@pytest.mark.parametrize("fmt,suffix", [("csv", ".csv"), ("html", ".html"), ("md", ".md")])
def test_reports_include_both_sections(tmp_path, fmt, suffix):
    file = tmp_path / "suite.json"
    file.write_text(json.dumps({
        "format_version": 1, "simulation": True,
        "character": {"name": "character", "scores": {"identity": .8}, "num_samples": 3},
        "temporal": {"name": "temporal", "scores": {"temporal": .7}, "num_samples": 2},
        "pbr": {"name": "pbr", "scores": {"pbr": .6}, "num_samples": 2}
    }))
    cmd_report(Namespace(results=str(file), format=fmt))
    out = file.with_suffix(suffix)
    content = out.read_text(encoding="utf-8")
    assert "character" in content.lower()
    assert "temporal" in content.lower()
    assert "pbr" in content.lower()
    assert out.stat().st_size > 50


def test_compare_accepts_native_suite_files(tmp_path, capsys):
    first = tmp_path / "first.json"
    other = tmp_path / "other.json"
    payload = {
        "character": {"name": "char", "scores": {"identity": .8}, "num_samples": 1},
        "temporal": {"name": "temp", "scores": {"temporal": .6}, "num_samples": 1},
        "pbr": {"name": "pbr", "scores": {"pbr": .6}, "num_samples": 1},
    }
    first.write_text(json.dumps(payload))
    payload["character"]["scores"]["identity"] = .9
    other.write_text(json.dumps(payload))
    cmd_compare(Namespace(baseline=str(first), qbp=str(other)))
    printed = capsys.readouterr().out
    assert "Section: character" in printed
    assert "Section: temporal" in printed
    assert "Section: pbr" in printed
    assert "+0.1000" in printed


def test_invalid_metric_fails_closed(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text(json.dumps({"name": "bad", "scores": {"identity": 9}, "num_samples": 1}))
    with pytest.raises(ValueError, match="invalid score"):
        load_results(file)


def test_html_report_escapes_untrusted_result_names():
    html = _suite_html({"result": BenchmarkResult(name="<script>alert(1)</script>",scores={"<img>": .5})})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img&gt;" in html


def test_invalid_sample_count_fails_closed(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text(json.dumps({"name": "bad", "scores": {"identity": .4}, "num_samples": -1}))
    with pytest.raises(ValueError, match="sample count"):
        load_results(file)


def test_existing_main_explicit_section_contract(tmp_path):
    from bench.cli import _load_result
    path = tmp_path / "combined.json"
    path.write_text(json.dumps({
        "kind": "synthetic-demo",
        "character": {"name": "character", "scores": {"identity": .7}, "num_samples": 2},
        "temporal": {"name": "temporal", "scores": {"temporal": .8}, "num_samples": 1},
    }))
    with pytest.raises(ValueError, match="--section"):
        _load_result(path)
    assert _load_result(path, "character").num_samples == 2
    cmd_report(Namespace(results=str(path), section="character", format="md"))
    assert (tmp_path / "combined-character.md").exists()
    assert not (tmp_path / "combined.md").exists()


def test_optional_section_comparison_preserves_full_suite_default(tmp_path, capsys):
    payload = {
        "character": {"name": "character", "scores": {"identity": .7}, "num_samples": 1},
        "temporal": {"name": "temporal", "scores": {"temporal": .8}, "num_samples": 1},
    }
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(payload))
    b.write_text(json.dumps(payload))
    cmd_compare(Namespace(baseline=str(a), qbp=str(b), section="character"))
    output = capsys.readouterr().out
    assert "Section: character" in output
    assert "Section: temporal" not in output
