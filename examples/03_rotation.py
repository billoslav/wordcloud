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

    # Mostly horizontal
    wc1 = Wordcloud(
        width=800,
        height=400,
        font_path=str(font_path),
        prefer_horizontal=0.9,
        rotation_angles=(90, -90),
        background_color="white",
    )
    wc1.generate(text, color_theme="viridis").draw_image(save_file=True, image_name="03_rotation_mostly_horizontal")
    print("Saved Results/03_rotation_mostly_horizontal.png")

    # More rotation
    wc2 = Wordcloud(
        width=800,
        height=400,
        font_path=str(font_path),
        prefer_horizontal=0.5,
        rotation_angles=(90, -90),
        background_color="white",
    )
    wc2.generate(text, color_theme="viridis").draw_image(save_file=True, image_name="03_rotation_more_vertical")
    print("Saved Results/03_rotation_more_vertical.png")


if __name__ == "__main__":
    main()


