"""Benchmark metrics for visual consistency evaluation.

Each metric function accepts typed inputs and returns a float in [0, 1].
Mock implementations use numpy/Pillow; swap with real model calls as needed.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from PIL import Image


def identity_similarity(
    embedding_a: np.ndarray | Sequence[float],
    embedding_b: np.ndarray | Sequence[float],
) -> float:
    """Cosine similarity between two identity embeddings.

    Args:
        embedding_a: First embedding vector.
        embedding_b: Second embedding vector.

    Returns:
        Similarity score in [0, 1].
    """
    a = np.asarray(embedding_a, dtype=np.float64)
    b = np.asarray(embedding_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def temporal_stability(frames: Sequence[Image.Image | np.ndarray]) -> float:
    """Measure visual stability across a sequence of frames.

    Computes mean absolute pixel difference between consecutive frames,
    normalized so that identical frames yield 1.0.

    Args:
        frames: Ordered sequence of images or arrays.

    Returns:
        Stability score in [0, 1] (1 = perfectly stable).
    """
    if len(frames) < 2:
        return 1.0
    diffs: list[float] = []
    for i in range(len(frames) - 1):
        img_a = _to_array(frames[i])
        img_b = _to_array(frames[i + 1])
        if img_a.shape != img_b.shape:
            diffs.append(1.0)
            continue
        mean_diff = float(np.mean(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))))
        normalized = mean_diff / 255.0
        diffs.append(1.0 - normalized)
    return float(np.mean(diffs))


def geometric_consistency(
    depth_a: np.ndarray | Image.Image,
    depth_b: np.ndarray | Image.Image,
) -> float:
    """Geometric consistency between two depth maps.

    Uses structural similarity on depth maps to measure alignment.

    Args:
        depth_a: First depth map.
        depth_b: Second depth map.

    Returns:
        Consistency score in [0, 1].
    """
    a = _to_depth_array(depth_a).astype(np.float64)
    b = _to_depth_array(depth_b).astype(np.float64)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    if a.size == 0:
        return 0.0
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mean_a) * (b - mean_b))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ssim = ((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / (
        (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(max(0.0, min(1.0, (ssim + 1.0) / 2.0)))


def aesthetic_score(image: Image.Image | np.ndarray) -> float:
    """Aesthetic quality assessment proxy.

    Uses color balance, contrast, and luminance variance as heuristics.
    Replace with a trained aesthetic predictor for production use.

    Args:
        image: Input image.

    Returns:
        Aesthetic score in [0, 1].
    """
    arr = _to_array(image).astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        lum = arr.mean(axis=2) if arr.ndim == 3 else arr.astype(np.float64)

    contrast = float(lum.std() / 128.0)
    contrast_score = min(1.0, contrast * 1.5)

    mean_lum = float(lum.mean())
    exposure_score = 1.0 - abs(mean_lum - 128.0) / 128.0

    color_sat = 0.0
    if arr.ndim == 3 and arr.shape[2] >= 3:
        hsv_max = arr.max(axis=2)
        hsv_min = arr.min(axis=2)
        saturation = np.where(hsv_max > 0, (hsv_max - hsv_min) / hsv_max, 0)
        color_sat = float(np.mean(saturation))

    score = 0.4 * contrast_score + 0.3 * exposure_score + 0.3 * color_sat
    return float(max(0.0, min(1.0, score)))


def pbr_compliance(texture_set: dict[str, Image.Image | np.ndarray]) -> float:
    """PBR material compliance check.

    Validates that albedo, roughness, and normal maps are consistent.
    Checks roughness values are in valid range, normal maps are unit-length,
    and albedo has reasonable luminance.

    Args:
        texture_set: Dict with keys 'albedo', 'roughness', 'normal'.

    Returns:
        Compliance score in [0, 1].
    """
    scores: list[float] = []

    if "roughness" in texture_set:
        rough = _to_array(texture_set["roughness"]).astype(np.float64)
        valid_range = np.all((rough >= 0) & (rough <= 255))
        mean_rough = rough.mean() / 255.0
        range_score = 1.0 if valid_range else 0.5
        distribution_score = 1.0 - abs(mean_rough - 0.5) * 2.0
        scores.append(0.5 * range_score + 0.5 * distribution_score)
    else:
        scores.append(0.0)

    if "normal" in texture_set:
        normal = _to_array(texture_set["normal"]).astype(np.float64)
        if normal.ndim == 3 and normal.shape[2] >= 3:
            n = normal[:, :, :3] / 255.0 * 2.0 - 1.0
            lengths = np.sqrt((n**2).sum(axis=2))
            unit_deviation = np.abs(lengths - 1.0)
            norm_score = 1.0 - float(np.mean(unit_deviation))
            scores.append(max(0.0, norm_score))
        else:
            scores.append(0.3)
    else:
        scores.append(0.0)

    if "albedo" in texture_set:
        albedo = _to_array(texture_set["albedo"]).astype(np.float64)
        if albedo.ndim == 3 and albedo.shape[2] >= 3:
            lum = 0.299 * albedo[:, :, 0] + 0.587 * albedo[:, :, 1] + 0.114 * albedo[:, :, 2]
        else:
            lum = albedo.mean(axis=2) if albedo.ndim == 3 else albedo.astype(np.float64)
        mean_lum = float(lum.mean()) / 255.0
        albedo_score = 1.0 - abs(mean_lum - 0.5) * 1.5
        scores.append(max(0.0, albedo_score))
    else:
        scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def overall_consistency(scores: dict[str, float]) -> float:
    """Weighted overall consistency from individual dimension scores.

    Weights:
        identity: 0.30
        temporal: 0.25
        geometric: 0.20
        aesthetic: 0.15
        pbr: 0.10

    Args:
        scores: Dict mapping dimension names to scores in [0, 1].

    Returns:
        Weighted overall score in [0, 1].
    """
    weights = {
        "identity": 0.30,
        "temporal": 0.25,
        "geometric": 0.20,
        "aesthetic": 0.15,
        "pbr": 0.10,
    }
    total_w = 0.0
    weighted_sum = 0.0
    for dim, w in weights.items():
        if dim in scores:
            weighted_sum += w * scores[dim]
            total_w += w
    if total_w < 1e-10:
        return 0.0
    return weighted_sum / total_w


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    return np.array(img)


def _to_depth_array(depth: np.ndarray | Image.Image) -> np.ndarray:
    arr = _to_array(depth)
    if arr.ndim == 3:
        return arr.mean(axis=2)
    return arr
