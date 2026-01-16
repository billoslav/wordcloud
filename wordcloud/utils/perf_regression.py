"""
Performance regression helpers for deterministic benchmarking.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..wordcloud import Wordcloud

DEFAULT_SEED = 42
DEFAULT_STRATEGIES = ["random", "rectangular", "archimedian"]
DEFAULT_TOLERANCE = 0.15
DEFAULT_RUNS = 3
DEFAULT_CONFIG = {
    "width": 800,
    "height": 500,
    "max_words": 200,
    "prefer_horizontal": 1.0,
    "rotation_angles": (90, -90),
}


@dataclass(frozen=True)
class PerfResult:
    strategy: str
    elapsed: float
    placed_words: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_benchmark_text(root: Optional[Path] = None) -> str:
    root = root or repo_root()
    lorem = root / "test_sources" / "lorem.txt"
    if lorem.exists():
        return lorem.read_text(encoding="utf-8")
    return "wordcloud benchmark " * 2000


def measure_strategy(
    strategy: str,
    text: str,
    seed: int,
    runs: int,
    config: Dict[str, object],
) -> PerfResult:
    font_path = str(repo_root() / "fonts" / "Arial Unicode.ttf")
    timings: List[float] = []
    placed_words = 0

    for _ in range(max(runs, 1)):
        wc = Wordcloud(
            width=int(config["width"]),
            height=int(config["height"]),
            font_path=font_path,
            place_strategy=strategy,
            max_words=int(config["max_words"]),
            prefer_horizontal=float(config["prefer_horizontal"]),
            rotation_angles=tuple(config["rotation_angles"]),
            random_state=seed,
        )
        start = time.perf_counter()
        wc.generate(text)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        placed_words = len(wc.gen_positions or [])

    return PerfResult(strategy=strategy, elapsed=min(timings), placed_words=placed_words)


def run_benchmark(
    strategies: Iterable[str],
    seed: int = DEFAULT_SEED,
    runs: int = DEFAULT_RUNS,
    config: Optional[Dict[str, object]] = None,
    text: Optional[str] = None,
) -> Dict[str, PerfResult]:
    config = config or DEFAULT_CONFIG
    text = text or load_benchmark_text()
    results = {}
    for strategy in strategies:
        results[strategy] = measure_strategy(strategy, text, seed, runs, config)
    return results


def serialize_results(results: Dict[str, PerfResult]) -> Dict[str, Dict[str, float]]:
    return {
        strategy: {
            "elapsed": result.elapsed,
            "placed_words": result.placed_words,
        }
        for strategy, result in results.items()
    }


def save_baseline(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_baseline(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_to_baseline(
    baseline: Dict[str, object],
    current: Dict[str, PerfResult],
    tolerance: float,
) -> List[str]:
    failures: List[str] = []
    baseline_results = baseline.get("results", {})

    for strategy, result in current.items():
        baseline_entry = baseline_results.get(strategy)
        if not baseline_entry:
            failures.append(f"Missing baseline for strategy '{strategy}'")
            continue
        baseline_time = float(baseline_entry["elapsed"])
        budget = baseline_time * (1 + tolerance)
        if result.elapsed > budget:
            failures.append(
                f"{strategy}: {result.elapsed:0.4f}s exceeds budget {budget:0.4f}s"
            )

    return failures
