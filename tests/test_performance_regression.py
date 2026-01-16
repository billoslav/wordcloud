"""
Optional performance regression checks (enabled via env var).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest # type: ignore

from wordcloud.utils.perf_regression import compare_to_baseline, load_baseline, run_benchmark

PERF_ENV = os.getenv("WORDCLOUD_PERF_BUDGETS") == "1"


@pytest.mark.performance
@pytest.mark.skipif(not PERF_ENV, reason="Set WORDCLOUD_PERF_BUDGETS=1 to enable perf checks")
def test_performance_regression() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_path = repo_root / "benchmarks" / "perf_baselines.json"
    baseline = load_baseline(baseline_path)

    seed = int(baseline.get("seed", 42))
    runs = int(baseline.get("runs", 1))
    tolerance = float(baseline.get("tolerance", 0.15))
    config = baseline.get("config", {})
    strategies = list(baseline.get("results", {}).keys())

    results = run_benchmark(strategies, seed=seed, runs=runs, config=config)
    failures = compare_to_baseline(baseline, results, tolerance)

    assert not failures, "Perf regression(s): " + "; ".join(failures)
