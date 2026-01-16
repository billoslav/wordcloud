"""
Benchmark rotation overhead.

Usage:
  python benchmarks/bench_rotation_overhead.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud


def load_text() -> str:
    lorem = repo_root / "test_sources" / "lorem.txt"
    if lorem.exists():
        return lorem.read_text(encoding="utf-8")
    return "rotation overhead benchmark " * 2000


def timed_generate(prefer_horizontal: float) -> float:
    font_path = str(repo_root / "fonts" / "Arial Unicode.ttf")
    seed = 42
    wc = Wordcloud(
        width=800,
        height=500,
        font_path=font_path,
        place_strategy="random",
        max_words=200,
        prefer_horizontal=prefer_horizontal,
        rotation_angles=(90, -90),
        random_state=seed,
    )
    start = time.perf_counter()
    wc.generate(load_text())
    return time.perf_counter() - start


def main() -> None:
    print("Benchmark: rotation overhead (single run each)")
    print("")
    for p in (1.0, 0.9, 0.7, 0.5):
        elapsed = timed_generate(p)
        print(f"prefer_horizontal={p:0.1f}  {elapsed:8.4f}s")


if __name__ == "__main__":
    main()


