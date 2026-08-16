# Latent Consistency Bench

Benchmarking visual consistency in AI-generated content.

## Overview

Latent Consistency Bench (LCB) provides standardized benchmarks comparing baseline generation against QBP-enhanced generation across multiple consistency dimensions:

- **Identity Consistency** - Character appearance stability across prompts
- **Temporal Consistency** - Frame-to-frame stability in sequential generation
- **Geometric Consistency** - Depth map alignment and spatial coherence
- **Aesthetic Quality** - Visual appeal and composition scoring
- **PBR Compliance** - Physically-based rendering material validity

## Installation

```bash
pip install -e .
# With ML dependencies:
pip install -e ".[ml]"
# With dev dependencies:
pip install -e ".[dev]"
```

## Quick Start

```python
from bench import ConsistencyBenchmark

# Define your generation functions
def baseline_generate(prompt: str) -> Image.Image:
    # Your baseline model
    ...

def qbp_generate(prompt: str) -> Image.Image:
    # Your QBP-enhanced model
    ...

# Run benchmarks
bench = ConsistencyBenchmark(baseline_generate, qbp_generate)

char_result = bench.run_character_consistency(n=20)
temp_result = bench.run_temporal_consistency(n=10)
pbr_result = bench.run_pbr_validity(n=50)

# Compare results
from bench.compare import compare_results
report = compare_results(char_result, temp_result)
print(report.to_markdown())
```

## CLI Usage

```bash
# Run full benchmark suite
lcb run --characters 20 --environments 10 --output results.json

# Compare two result files
lcb compare baseline.json qbp.json

# Generate report in different formats
lcb report results.json --format html
lcb report results.json --format csv
lcb report results.json --format md
```

## Benchmark Methodology

### Character Consistency
Tests identity preservation across N subjects with varying poses and styles. Each subject generates prompts combining style and pose variations, then measures:
- Identity embedding cosine similarity between baseline and QBP outputs
- Aesthetic quality of both outputs
- Geometric consistency of generated depth maps

### Temporal Consistency
Tests animation stability by generating sequential frames (5 frames per sequence) for N motion subjects. Measures frame-to-frame pixel stability using normalized mean absolute difference.

### PBR Validity
Tests physically-based rendering compliance across N material samples. Validates texture sets (albedo, roughness, normal maps) for:
- Roughness values in valid range
- Normal map unit-length vectors
- Albedo luminance distribution

### Scoring

All metrics return values in [0, 1] where 1 is optimal. The overall score uses weighted averaging:

| Dimension | Weight |
|-----------|--------|
| Identity | 0.30 |
| Temporal | 0.25 |
| Geometric | 0.20 |
| Aesthetic | 0.15 |
| PBR | 0.10 |

## Project Structure

```
latent-consistency-bench/
├── bench/
│   ├── __init__.py          # Package exports
│   ├── benchmark.py         # Main ConsistencyBenchmark runner
│   ├── metrics.py           # Individual metric implementations
│   ├── test_cases.py        # Standardized test case definitions
│   ├── compare.py           # Comparison and visualization tools
│   ├── report.py            # Result dataclasses and formatting
│   ├── cli.py               # Command-line interface
│   └── datasets/            # Sample test data
│       ├── sample_characters.json
│       └── sample_environments.json
├── tests/
│   ├── conftest.py          # Shared pytest fixtures
│   ├── test_benchmark.py    # Benchmark runner tests
│   └── test_metrics.py      # Metric function tests
├── pyproject.toml
├── LICENSE                  # Apache 2.0
└── README.md
```

## Mock Implementations

This library provides mock implementations for heavy ML dependencies (identity embeddings, depth maps, texture generation). The interfaces are designed for easy swapping with real model calls:

```python
# Replace mock embedding with real model
def real_embedding(image: Image.Image) -> np.ndarray:
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    return model.get_image_features(**inputs).detach().numpy().flatten()
```

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
