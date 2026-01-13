from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud
from wordcloud.utils.export import REPORTLAB_AVAILABLE, PIL_AVAILABLE, PDFExporter, AnimatedGIFExporter

from _text_sources import load_example_text


def main() -> None:
    results = repo_root / "Results"
    results.mkdir(parents=True, exist_ok=True)

    font_path = repo_root / "fonts" / "Arial Unicode.ttf"
    wc = Wordcloud(width=800, height=400, font_path=str(font_path), background_color="white")
    wc.generate(load_example_text("lorem.txt"), color_theme="viridis")
    img = wc.draw_image(save_file=False)

    # PDF (ReportLab)
    if REPORTLAB_AVAILABLE and PIL_AVAILABLE:
        PDFExporter().export_pdf(img, results / "06_wordcloud.pdf", title="Wordcloud PDF Export")
        print("Saved Results/06_wordcloud.pdf")
    else:
        print("PDF export skipped. Install extra: reportlab")

    # GIF (Pillow)
    if PIL_AVAILABLE:
        frames = [img] * 5
        AnimatedGIFExporter().export_animated_gif(frames, results / "06_wordcloud.gif", duration=120)
        print("Saved Results/06_wordcloud.gif")
    else:
        print("GIF export skipped. Install extra: pillow")


if __name__ == "__main__":
    main()


