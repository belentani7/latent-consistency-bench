"""Guarantee benchmarks do not mutate host applications' random streams."""
import random

import numpy as np
from PIL import Image

from bench.benchmark import ConsistencyBenchmark


def test_constructing_benchmark_preserves_python_and_numpy_rng():
    random.seed(2026)
    np.random.seed(2026)
    before_python = random.getstate()
    before_numpy = np.random.get_state()

    generate = lambda prompt: Image.new("RGB", (16, 16), (21, 42, 84))
    ConsistencyBenchmark(generate, generate, seed=321)

    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    assert np.array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]


def test_running_synthetic_character_suite_does_not_advance_global_rng():
    random.seed(2026)
    np.random.seed(2026)
    before_python = random.getstate()
    before_numpy = np.random.get_state()

    generate = lambda prompt: Image.new("RGB", (16, 16), (21, 42, 84))
    benchmark = ConsistencyBenchmark(generate, generate, seed=123)
    outcome = benchmark.run_character_consistency(n=1)

    assert outcome.num_samples > 0
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    assert np.array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]
