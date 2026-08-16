"""Shared fixtures for latent-consistency-bench tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """A deterministic 64x64 RGB test image."""
    rng = np.random.RandomState(42)
    return Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8))


@pytest.fixture
def sample_images() -> list[Image.Image]:
    """Three deterministic 64x64 RGB test images."""
    images = []
    for seed in [42, 43, 44]:
        rng = np.random.RandomState(seed)
        images.append(Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)))
    return images


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """A normalized 128-d identity embedding."""
    rng = np.random.RandomState(42)
    emb = rng.randn(128)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def sample_depth() -> np.ndarray:
    """A 64x64 mock depth map."""
    rng = np.random.RandomState(42)
    return rng.rand(64, 64).astype(np.float64) * 10.0


@pytest.fixture
def sample_texture_set() -> dict[str, Image.Image]:
    """A mock PBR texture set with albedo, roughness, and normal."""
    rng = np.random.RandomState(42)
    albedo = Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    roughness = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode="L")
    normal_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    normal_arr[:, :, 2] = 255
    normal = Image.fromarray(normal_arr)
    return {"albedo": albedo, "roughness": roughness, "normal": normal}


@pytest.fixture
def mock_baseline():
    """Mock baseline generation function."""
    def generate(prompt: str, **kwargs) -> Image.Image:
        rng = np.random.RandomState(hash(prompt) % (2**31))
        return Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    return generate


@pytest.fixture
def mock_qbp():
    """Mock QBP-enhanced generation function."""
    def generate(prompt: str, **kwargs) -> Image.Image:
        rng = np.random.RandomState(hash(prompt) % (2**31) + 1)
        return Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    return generate
