"""Tests for bench/benchmark.py."""

from __future__ import annotations

import json
import pytest
from PIL import Image

from bench.benchmark import ConsistencyBenchmark
from bench.report import BenchmarkResult
from bench.test_cases import (
    CharacterTestCase,
    EnvironmentTestCase,
    ProductTestCase,
    default_character_cases,
)


class TestConsistencyBenchmark:
    def test_init(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp)
        assert bench.seed == 42

    def test_run_with_character_cases(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        cases = default_character_cases(n=2)
        result = bench.run(cases, name="test_run")
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_run"
        assert result.num_samples > 0
        assert "identity" in result.scores
        assert "temporal" in result.scores
        assert "geometric" in result.scores
        assert "aesthetic" in result.scores
        assert "pbr" in result.scores

    def test_run_character_consistency(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        result = bench.run_character_consistency(n=3)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "character_consistency"
        assert result.num_samples > 0

    def test_run_temporal_consistency(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        result = bench.run_temporal_consistency(n=3)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "temporal_consistency"
        assert "temporal" in result.scores

    def test_run_pbr_validity(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        result = bench.run_pbr_validity(n=5)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "pbr_validity"
        assert "pbr" in result.scores

    def test_reproducibility(self, mock_baseline, mock_qbp):
        bench1 = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        bench2 = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        cases = default_character_cases(n=2)
        r1 = bench1.run(cases)
        r2 = bench2.run(cases)
        assert r1.scores == r2.scores

    def test_overall_in_metadata(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        cases = default_character_cases(n=2)
        result = bench.run(cases)
        assert "overall" in result.metadata

    def test_result_to_json(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        cases = default_character_cases(n=2)
        result = bench.run(cases)
        json_str = result.to_json()
        data = json.loads(json_str)
        assert "scores" in data
        assert "overall" in data

    def test_result_to_markdown(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        cases = default_character_cases(n=2)
        result = bench.run(cases)
        md = result.to_markdown()
        assert "# Benchmark Result:" in md
        assert "Dimension" in md

    def test_result_to_html(self, mock_baseline, mock_qbp):
        bench = ConsistencyBenchmark(mock_baseline, mock_qbp, seed=42)
        cases = default_character_cases(n=2)
        result = bench.run(cases)
        html = result.to_html()
        assert "<!DOCTYPE html>" in html
        assert result.name in html


class TestBenchmarkResult:
    def test_overall_property(self):
        r = BenchmarkResult(name="test", scores={"a": 0.8, "b": 0.6})
        assert r.overall == pytest.approx(0.7, abs=1e-6)

    def test_overall_empty(self):
        r = BenchmarkResult(name="test")
        assert r.overall == 0.0

    def test_to_dict(self):
        r = BenchmarkResult(name="test", scores={"a": 0.5}, num_samples=10)
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["num_samples"] == 10
        assert "overall" in d
