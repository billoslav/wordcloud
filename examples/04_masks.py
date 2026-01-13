from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud

from _text_sources import load_example_text


def circle_mask(size: int = 400) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    cy = size // 2
    cx = size // 2
    r = size // 2 - 10
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).astype(np.uint8)


def main() -> None:
    font_path = repo_root / "fonts" / "Arial Unicode.ttf"

    mask = circle_mask(400)
    text = load_example_text("generated.txt")

    wc = Wordcloud(
        width=mask.shape[1],
        height=mask.shape[0],
        font_path=str(font_path),
        mask_image=mask,
        background_color="white",
        prefer_horizontal=0.8,
    )
    wc.generate(text, color_theme="plasma").draw_image(save_file=True, image_name="04_mask_circle")
    print("Saved Results/04_mask_circle.png")


if __name__ == "__main__":
    main()


