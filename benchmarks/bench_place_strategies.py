"""
Benchmark word placement strategies.

Usage:
  python benchmarks/bench_place_strategies.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud
from wordcloud.utils import STRATEGIES


def load_text() -> str:
    lorem = repo_root / "test_sources" / "lorem.txt"
    if lorem.exists():
        return lorem.read_text(encoding="utf-8")
    return "wordcloud benchmark " * 2000


def main() -> None:
    text = load_text()
    font_path = str(repo_root / "fonts" / "Arial Unicode.ttf")
    seed = 42

    print("Benchmark: placement strategies")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print("")

    for strategy in STRATEGIES:
        wc = Wordcloud(
            width=800,
            height=500,
            font_path=font_path,
            place_strategy=strategy,
            max_words=200,
            prefer_horizontal=1.0,
            random_state=seed,
        )

        start = time.perf_counter()
        wc.generate(text)
        elapsed = time.perf_counter() - start

        placed = len(wc.gen_positions or [])
        print(f"{strategy:18s}  {elapsed:8.4f}s  placed_words={placed}")


if __name__ == "__main__":
    main()


