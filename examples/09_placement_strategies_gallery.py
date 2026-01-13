from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud
from wordcloud.utils import STRATEGIES

from _text_sources import load_example_text


def main() -> None:
    font_path = repo_root / "fonts" / "Arial Unicode.ttf"

    text = load_example_text("lorem.txt")

    for strategy in STRATEGIES:
        wc = Wordcloud(
            width=800,
            height=400,
            font_path=str(font_path),
            place_strategy=strategy,
            background_color="white",
            prefer_horizontal=1.0,
        )
        wc.generate(text, color_theme="viridis")
        wc.draw_image(save_file=True, image_name=f"09_strategy_{strategy}")
        print(f"Saved Results/09_strategy_{strategy}.png")


if __name__ == "__main__":
    main()


