# LATENT CONSISTENCY BENCH
**Visual quality is measurable only when the measurement is honest.**

Latent Consistency Bench (LCB) is a Python toolkit for **consistency analysis of generated images**: character identity, sequential frame stability, geometric coherence, appearance heuristics and PBR texture validity. It includes benchmark interfaces, result models, comparisons, charts and reporting.

> **Truth in measurements:** the built-in `lcb run` command generates deterministic **synthetic noise**, not images from a real generative model. It exists to exercise the end-to-end pipeline and reporting contract. Identity embeddings, depth proxies and PBR materials are mock implementations. **Its numbers cannot be used to claim real model quality, superiority or scientific validity.** Connect genuine generation and evaluation functions before performing real evaluations.

## Quick start — reproducible offline simulation

Requires Python 3.10+.

```bash
git clone https://github.com/belentani7/latent-consistency-bench.git
cd latent-consistency-bench
python -m pip install -e ".[dev]"
lcb run --characters 3 --environments 2 --materials 3 --seed 42 --output artifacts/demo.json
lcb report artifacts/demo.json --format html
lcb report artifacts/demo.json --format md
lcb report artifacts/demo.json --format csv
lcb compare artifacts/demo.json artifacts/demo.json
```

Output:
- `artifacts/demo.json`: native, versioned suite with `character`, `temporal` and `pbr` sections, each with scores, metadata and sample count; `simulation: true` marks the origin.
- `artifacts/demo.png`: bar chart of those three synthetic suites.
- `artifacts/demo.{html,md,csv}`: reports that now consume the **same JSON schema emitted by `run`**.
- `lcb compare A.json B.json`: compares each common section, preserving earlier single-result JSON support. Unmatched sections are identified, never silently interpreted as zeros.

`--seed` uses SHA-256-derived deterministic image seeds, independent of Python's randomized `hash()` salts. Running twice with identical inputs produces the same JSON scores on the same compatible stack.

## Evaluating real generators

Instead of the included synthetic generators, provide the generation callables yourself:

```python
from bench import ConsistencyBenchmark

def baseline_generate(prompt: str):
    # Return a real PIL.Image.Image for this prompt.
    raise NotImplementedError("Connect a real provider")

def candidate_generate(prompt: str):
    raise NotImplementedError("Connect a second generator")

benchmark = ConsistencyBenchmark(baseline_generate, candidate_generate, seed=42)
result = benchmark.run_character_consistency(n=20)
print(result.to_markdown())
```

**The default feature extractors are still proxies**, so this alone does not turn the toolkit into a validated scientific benchmark. Replace the embedding, depth and material heuristics with suitable real evaluators, maintain comparable prompts, seeding, model settings, and report confidence intervals and experimental limits.

## Dimensions

| Dimension | Meaning | Limit of current implementation |
|---|---|---|
| Identity | Cosine similarity of extracted embeddings | Synthetic feature extractor, not a trained identity model |
| Temporal | Frame-to-frame pixel difference | Sensitive to legitimate scene and camera motion |
| Geometric | Depth-map structural comparison | Luminance-based depth proxy |
| Aesthetic | Exposure/contrast/color heuristic | Not a human preference or trained aesthetic model |
| PBR | Texture map distribution and normals | Synthetic maps, not physical proof |

Weights used by `overall_consistency` are identity 30%, temporal 25%, geometric 20%, aesthetic 15%, PBR 10%. `BenchmarkResult.overall` is an **unweighted arithmetic mean of present dimensions**, and should not be conflated with the weighted formula.

## Commands

```text
lcb run --characters N --environments N --materials N --seed N --output file.json
lcb compare baseline.json candidate.json
lcb report results.json --format html|csv|md
```

## Development and quality gates

```bash
MPLBACKEND=Agg python -m pytest -q
python -m compileall -q bench tests
```

The suite covers consistency dimensions, metric bounds, reproducible simulation, report JSON round-trip, native comparison, HTML/Markdown/CSV export and escaping of result names. The GitHub CI must actually execute successfully before a release is promoted.

## Design and licensing

[Static project site](docs/index.html), [Python API](bench/benchmark.py), [CLI](bench/cli.py), [report model](bench/report.py), [sample datasets](bench/datasets). No files, methods or previous capabilities have been removed. Apache 2.0 — see LICENSE.
