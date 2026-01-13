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

    text = load_example_text("lorem.txt")

    wc = Wordcloud(width=800, height=400, font_path=str(font_path), background_color="white")
    wc.generate(text, color_theme="viridis")
    wc.draw_image(save_file=True, image_name="00_quickstart")

    print("Saved Results/00_quickstart.png")


if __name__ == "__main__":
    main()


