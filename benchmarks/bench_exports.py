"""
Benchmark exports (PNG, SVG, HTML) and optionally PDF/GIF if dependencies exist.

Usage:
  python benchmarks/bench_exports.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud
from wordcloud.utils.export import REPORTLAB_AVAILABLE, PIL_AVAILABLE, AnimatedGIFExporter, PDFExporter


def load_text() -> str:
    lorem = repo_root / "test_sources" / "lorem.txt"
    if lorem.exists():
        return lorem.read_text(encoding="utf-8")
    return "export benchmark " * 2000


def main() -> None:
    results = repo_root / "Results"
    results.mkdir(parents=True, exist_ok=True)

    font_path = str(repo_root / "fonts" / "Arial Unicode.ttf")
    wc = Wordcloud(width=800, height=500, font_path=font_path, max_words=200, prefer_horizontal=1.0)

    t0 = time.perf_counter()
    wc.generate(load_text(), color_theme="viridis")
    t_gen = time.perf_counter() - t0

    print("Benchmark: exports")
    print(f"generate(): {t_gen:0.4f}s")

    # PNG
    t0 = time.perf_counter()
    img = wc.draw_image(save_file=False)
    img.save(results / "bench.png")
    print(f"PNG save:  {time.perf_counter() - t0:0.4f}s")

    # SVG
    t0 = time.perf_counter()
    svg = wc.generate_svg(save_file=True, file_name="bench")
    _ = svg
    print(f"SVG gen:   {time.perf_counter() - t0:0.4f}s")

    # HTML
    t0 = time.perf_counter()
    wc.create_html(save_file=True, file_name="bench")
    print(f"HTML gen:  {time.perf_counter() - t0:0.4f}s")

    # PDF (optional)
    if REPORTLAB_AVAILABLE and PIL_AVAILABLE:
        t0 = time.perf_counter()
        PDFExporter().export_pdf(img, results / "bench.pdf")
        print(f"PDF save:  {time.perf_counter() - t0:0.4f}s")
    else:
        print("PDF save:  skipped (install reportlab)")

    # GIF (optional)
    if PIL_AVAILABLE:
        t0 = time.perf_counter()
        frames = [img] * 3
        AnimatedGIFExporter().export_animated_gif(frames, results / "bench.gif", duration=80)
        print(f"GIF save:  {time.perf_counter() - t0:0.4f}s")
    else:
        print("GIF save:  skipped (install pillow)")


if __name__ == "__main__":
    main()


