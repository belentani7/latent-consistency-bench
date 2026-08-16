"""Main benchmark runner for visual consistency evaluation."""

from __future__ import annotations

import random
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from bench.metrics import (
    aesthetic_score,
    geometric_consistency,
    identity_similarity,
    overall_consistency,
    pbr_compliance,
    temporal_stability,
)
from bench.report import BenchmarkResult
from bench.test_cases import (
    CharacterTestCase,
    EnvironmentTestCase,
    ProductTestCase,
    default_character_cases,
    default_environment_cases,
    default_product_cases,
)


class GenerationFn(Protocol):
    """Protocol for image generation functions.

    A generation function takes a text prompt and returns an image.
    """

    def __call__(self, prompt: str, **kwargs: Any) -> Image.Image: ...


class ConsistencyBenchmark:
    """Benchmark runner for visual consistency evaluation.

    Compares a baseline generation function against a QBP-enhanced variant
    across multiple consistency dimensions.

    Args:
        baseline_fn: Standard generation function.
        qbp_fn: QBP-enhanced generation function.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        baseline_fn: GenerationFn,
        qbp_fn: GenerationFn,
        seed: int = 42,
    ) -> None:
        self.baseline_fn = baseline_fn
        self.qbp_fn = qbp_fn
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def run(
        self,
        test_cases: list[CharacterTestCase | EnvironmentTestCase | ProductTestCase],
        *,
        name: str = "benchmark",
    ) -> BenchmarkResult:
        """Run a full benchmark suite across all test cases.

        Args:
            test_cases: List of test case objects.
            name: Result name label.

        Returns:
            BenchmarkResult with per-dimension scores.
        """
        all_scores: dict[str, list[float]] = {
            "identity": [],
            "temporal": [],
            "geometric": [],
            "aesthetic": [],
            "pbr": [],
        }
        total_samples = 0

        for case in test_cases:
            prompts = self._extract_prompts(case)
            if not prompts:
                continue

            for prompt in prompts:
                try:
                    img_baseline = self.baseline_fn(prompt)
                    img_qbp = self.qbp_fn(prompt)
                except Exception:
                    continue

                total_samples += 1

                emb_b = self._mock_embedding(img_baseline)
                emb_q = self._mock_embedding(img_qbp)
                all_scores["identity"].append(identity_similarity(emb_b, emb_q))

                all_scores["aesthetic"].append(
                    0.5 * (aesthetic_score(img_baseline) + aesthetic_score(img_qbp))
                )

                frames = [img_baseline, img_qbp]
                all_scores["temporal"].append(temporal_stability(frames))

                depth_b = self._mock_depth(img_baseline)
                depth_q = self._mock_depth(img_qbp)
                all_scores["geometric"].append(geometric_consistency(depth_b, depth_q))

                tex_b = self._mock_textures(img_baseline)
                tex_q = self._mock_textures(img_qbp)
                all_scores["pbr"].append(
                    0.5 * (pbr_compliance(tex_b) + pbr_compliance(tex_q))
                )

        averaged = {k: float(np.mean(v)) if v else 0.0 for k, v in all_scores.items()}
        return BenchmarkResult(
            name=name,
            scores=averaged,
            metadata={
                "seed": self.seed,
                "num_test_cases": len(test_cases),
                "overall": overall_consistency(averaged),
            },
            num_samples=total_samples,
        )

    def run_character_consistency(self, n: int = 20) -> BenchmarkResult:
        """Run character consistency benchmark.

        Args:
            n: Number of character subjects to test.

        Returns:
            BenchmarkResult for character consistency.
        """
        cases = default_character_cases(n)
        return self.run(cases, name="character_consistency")

    def run_temporal_consistency(self, n: int = 10) -> BenchmarkResult:
        """Run temporal consistency benchmark.

        Generates sequential frames for animation stability testing.

        Args:
            n: Number of temporal sequences to test.

        Returns:
            BenchmarkResult for temporal consistency.
        """
        subjects = [
            "character walking", "car driving", "bird flying",
            "water flowing", "clouds moving", "leaves falling",
            "waves crashing", "candle flickering", "flag waving",
            "fire burning",
        ][:n]

        scores: list[float] = []
        for subject in subjects:
            frames = []
            for i in range(5):
                prompt = f"{subject}, frame {i}"
                try:
                    frames.append(self.qbp_fn(prompt))
                except Exception:
                    continue
            if len(frames) >= 2:
                scores.append(temporal_stability(frames))

        score = float(np.mean(scores)) if scores else 0.0
        return BenchmarkResult(
            name="temporal_consistency",
            scores={"temporal": score},
            metadata={"num_sequences": n},
            num_samples=len(scores),
        )

    def run_pbr_validity(self, n: int = 50) -> BenchmarkResult:
        """Run PBR material validity benchmark.

        Args:
            n: Number of PBR texture sets to test.

        Returns:
            BenchmarkResult for PBR compliance.
        """
        materials = [
            "brushed metal", "polished wood", "rough stone",
            "smooth plastic", "woven fabric", "matte ceramic",
            "glossy paint", "raw leather", "translucent glass",
            "oxidized copper",
        ]

        scores: list[float] = []
        for i in range(n):
            material = materials[i % len(materials)]
            prompt = f"PBR texture set for {material}"
            try:
                img = self.qbp_fn(prompt)
                tex_set = self._mock_textures(img)
                scores.append(pbr_compliance(tex_set))
            except Exception:
                continue

        score = float(np.mean(scores)) if scores else 0.0
        return BenchmarkResult(
            name="pbr_validity",
            scores={"pbr": score},
            metadata={"num_materials": n},
            num_samples=len(scores),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_prompts(
        self, case: CharacterTestCase | EnvironmentTestCase | ProductTestCase
    ) -> list[str]:
        return case.prompts()

    def _mock_embedding(self, img: Image.Image) -> np.ndarray:
        """Generate a deterministic mock embedding from image content."""
        arr = np.array(img).astype(np.float64)
        if arr.ndim == 3:
            mean_colors = arr.mean(axis=(0, 1))
            rng = np.random.RandomState(int(arr.mean()) % (2**31))
            emb = rng.randn(128)
            emb[:3] = mean_colors / 255.0
        else:
            rng = np.random.RandomState(int(arr.mean()) % (2**31))
            emb = rng.randn(128)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _mock_depth(self, img: Image.Image) -> np.ndarray:
        """Generate a mock depth map from image luminance."""
        arr = np.array(img).astype(np.float64)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        else:
            lum = arr.mean(axis=2) if arr.ndim == 3 else arr.astype(np.float64)
        return (lum / 255.0 * 10.0)

    def _mock_textures(self, img: Image.Image) -> dict[str, Image.Image]:
        """Generate mock PBR texture set from a single image."""
        arr = np.array(img)
        albedo = Image.fromarray(arr)
        h, w = arr.shape[:2]
        roughness_arr = np.full((h, w), 128, dtype=np.uint8)
        roughness = Image.fromarray(roughness_arr, mode="L")
        normal_arr = np.zeros((h, w, 3), dtype=np.uint8)
        normal_arr[:, :, 2] = 255
        normal = Image.fromarray(normal_arr)
        return {"albedo": albedo, "roughness": roughness, "normal": normal}
