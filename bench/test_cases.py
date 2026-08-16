"""Standardized test cases for consistency benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STANDARD_SUBJECTS: list[str] = [
    "portrait of a young woman",
    "portrait of an elderly man",
    "full-body athlete in motion",
    "child playing in a park",
    "group of friends at a cafe",
    "robot in a workshop",
    "cat sitting on a windowsill",
    "dog running on a beach",
    "fantasy warrior with armor",
    "astronaut on Mars surface",
    "medieval castle at sunset",
    "modern city skyline at night",
    "underwater coral reef scene",
    "snowy mountain landscape",
    "tropical jungle with waterfalls",
    "steampunk airship in clouds",
    "ancient Greek temple ruins",
    "neon-lit cyberpunk alley",
    "cozy cabin in autumn forest",
    "abstract geometric composition",
]

STANDARD_STYLES: list[str] = [
    "photorealistic",
    "oil painting",
    "watercolor",
    "anime",
    "pixel art",
    "3D render",
    "pencil sketch",
    "pop art",
    "impressionist",
    "minimalist",
]

STANDARD_POSES: list[str] = [
    "front facing",
    "three-quarter view",
    "profile left",
    "profile right",
    "looking up",
    "looking down",
    "arms crossed",
    "hands on hips",
    "walking",
    "running",
    "sitting",
    "kneeling",
    "jumping",
    "reaching forward",
    "turning around",
]


@dataclass
class CharacterTestCase:
    """Test case for character consistency evaluation.

    Attributes:
        subject: Text prompt describing the character.
        poses: List of pose variations to test.
        styles: List of style variations to test.
        reference_embedding: Optional pre-computed identity embedding.
    """

    subject: str
    poses: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=lambda: ["photorealistic"])
    reference_embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.poses:
            self.poses = STANDARD_POSES[:5]

    def prompts(self) -> list[str]:
        prompts = []
        for style in self.styles:
            for pose in self.poses:
                prompts.append(f"{self.subject}, {style} style, {pose}")
        return prompts


@dataclass
class EnvironmentTestCase:
    """Test case for environment consistency evaluation.

    Attributes:
        subject: Text prompt describing the environment.
        lighting: List of lighting conditions.
        weather: List of weather/time-of-day conditions.
    """

    subject: str
    lighting: list[str] = field(default_factory=lambda: [
        "golden hour", "midday", "night", "overcast", "studio lighting",
    ])
    weather: list[str] = field(default_factory=lambda: [
        "clear", "cloudy", "rainy", "foggy", "snowy",
    ])

    def prompts(self) -> list[str]:
        prompts = []
        for light in self.lighting:
            for w in self.weather:
                prompts.append(f"{self.subject}, {light}, {w} weather")
        return prompts


@dataclass
class ProductTestCase:
    """Test case for product rendering consistency.

    Attributes:
        subject: Text prompt describing the product.
        angles: List of camera angles.
        backgrounds: List of background descriptions.
    """

    subject: str
    angles: list[str] = field(default_factory=lambda: [
        "front", "back", "left side", "right side", "top-down", "45-degree",
    ])
    backgrounds: list[str] = field(default_factory=lambda: [
        "white studio", "gradient gray", "lifestyle context", "transparent",
    ])

    def prompts(self) -> list[str]:
        prompts = []
        for angle in self.angles:
            for bg in self.backgrounds:
                prompts.append(f"{self.subject}, {angle} view, {bg} background")
        return prompts


def default_character_cases(n: int = 5) -> list[CharacterTestCase]:
    """Generate n default character test cases from standard subjects."""
    cases = []
    for i in range(min(n, len(STANDARD_SUBJECTS))):
        cases.append(CharacterTestCase(
            subject=STANDARD_SUBJECTS[i],
            poses=STANDARD_POSES[:5],
            styles=["photorealistic", "oil painting"],
        ))
    return cases


def default_environment_cases(n: int = 3) -> list[EnvironmentTestCase]:
    """Generate n default environment test cases."""
    subjects = [
        "modern living room interior",
        "city street with pedestrians",
        "natural forest clearing",
        "space station corridor",
        "underwater cave system",
    ]
    return [EnvironmentTestCase(subject=s) for s in subjects[:n]]


def default_product_cases(n: int = 3) -> list[ProductTestCase]:
    """Generate n default product test cases."""
    subjects = [
        "designer wristwatch",
        "smartphone",
        "running shoe",
        "coffee mug",
        "backpack",
    ]
    return [ProductTestCase(subject=s) for s in subjects[:n]]
