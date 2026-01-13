from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud

from _text_sources import load_example_text


def main() -> None:
    font_path = repo_root / "fonts" / "Arial Unicode.ttf"

    wc = Wordcloud(
        width=800,
        height=400,
        font_path=str(font_path),
        enable_performance_tracking=True,
        performance_tracking_detail="detailed",
        background_color="white",
    )
    wc.generate(load_example_text("generated.txt"), color_theme="viridis")
    wc.draw_image(save_file=True, image_name="08_performance_tracking")

    print("Saved Results/08_performance_tracking.png")
    print("")
    print("Performance metrics:")
    for key, value in wc.performance_metrics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()


