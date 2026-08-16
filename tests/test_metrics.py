"""Tests for bench/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bench.metrics import (
    aesthetic_score,
    geometric_consistency,
    identity_similarity,
    overall_consistency,
    pbr_compliance,
    temporal_stability,
)


class TestIdentitySimilarity:
    def test_identical_vectors(self, sample_embedding):
        score = identity_similarity(sample_embedding, sample_embedding)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        score = identity_similarity(a, b)
        assert 0.4 <= score <= 0.6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        score = identity_similarity(a, b)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self):
        a = np.zeros(10)
        b = np.ones(10)
        score = identity_similarity(a, b)
        assert score == 0.0

    def test_range_output(self):
        rng = np.random.RandomState(42)
        a, b = rng.randn(64), rng.randn(64)
        score = identity_similarity(a, b)
        assert 0.0 <= score <= 1.0


class TestTemporalStability:
    def test_identical_frames(self, sample_image):
        frames = [sample_image, sample_image]
        score = temporal_stability(frames)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_single_frame(self, sample_image):
        score = temporal_stability([sample_image])
        assert score == 1.0

    def test_empty_frames(self):
        score = temporal_stability([])
        assert score == 1.0

    def test_different_frames(self, sample_images):
        score = temporal_stability(sample_images)
        assert 0.0 <= score <= 1.0

    def test_range_output(self, sample_images):
        score = temporal_stability(sample_images)
        assert isinstance(score, float)


class TestGeometricConsistency:
    def test_identical_depths(self, sample_depth):
        score = geometric_consistency(sample_depth, sample_depth)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_different_depths(self):
        rng = np.random.RandomState(42)
        a = rng.rand(64, 64) * 10.0
        b = rng.rand(64, 64) * 10.0
        score = geometric_consistency(a, b)
        assert 0.0 <= score <= 1.0

    def test_pil_images(self):
        rng = np.random.RandomState(42)
        a = Image.fromarray((rng.rand(32, 32) * 255).astype(np.uint8), mode="L")
        b = Image.fromarray((rng.rand(32, 32) * 255).astype(np.uint8), mode="L")
        score = geometric_consistency(a, b)
        assert 0.0 <= score <= 1.0

    def test_mismatched_shapes(self):
        rng = np.random.RandomState(42)
        a = rng.rand(32, 32)
        b = rng.rand(64, 64)
        score = geometric_consistency(a, b)
        assert 0.0 <= score <= 1.0


class TestAestheticScore:
    def test_returns_float(self, sample_image):
        score = aesthetic_score(sample_image)
        assert isinstance(score, float)

    def test_range(self, sample_image):
        score = aesthetic_score(sample_image)
        assert 0.0 <= score <= 1.0

    def test_grayscale(self):
        arr = np.full((32, 32), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        score = aesthetic_score(img)
        assert 0.0 <= score <= 1.0


class TestPbrCompliance:
    def test_valid_texture_set(self, sample_texture_set):
        score = pbr_compliance(sample_texture_set)
        assert 0.0 <= score <= 1.0

    def test_missing_roughness(self, sample_texture_set):
        del sample_texture_set["roughness"]
        score = pbr_compliance(sample_texture_set)
        assert 0.0 <= score <= 1.0

    def test_empty_texture_set(self):
        score = pbr_compliance({})
        assert score == 0.0

    def test_normal_map_only(self, sample_texture_set):
        score = pbr_compliance({"normal": sample_texture_set["normal"]})
        assert 0.0 <= score <= 1.0


class TestOverallConsistency:
    def test_all_present(self):
        scores = {
            "identity": 0.9,
            "temporal": 0.8,
            "geometric": 0.7,
            "aesthetic": 0.85,
            "pbr": 0.6,
        }
        overall = overall_consistency(scores)
        assert 0.0 <= overall <= 1.0
        assert overall == pytest.approx(0.9 * 0.30 + 0.8 * 0.25 + 0.7 * 0.20 + 0.85 * 0.15 + 0.6 * 0.10, abs=1e-6)

    def test_partial_scores(self):
        scores = {"identity": 1.0, "temporal": 1.0}
        overall = overall_consistency(scores)
        assert 0.0 <= overall <= 1.0

    def test_empty_scores(self):
        overall = overall_consistency({})
        assert overall == 0.0

    def test_weighted_average(self):
        scores = {"identity": 1.0}
        overall = overall_consistency(scores)
        assert overall == pytest.approx(1.0, abs=1e-6)
