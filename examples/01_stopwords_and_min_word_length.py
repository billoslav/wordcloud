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

    base = load_example_text("generated.txt")
    text = "The THE the and AND and is IS is " + base

    # Note: stopwords are checked case-sensitively before lowercasing.
    # That means "The" is NOT removed by stopwords=["the"], and becomes "the".
    wc = Wordcloud(
        width=900,
        height=450,
        font_path=str(font_path),
        stopwords=["the", "and", "is"],
        min_word_length=3,
        background_color="white",
    )
    wc.generate(text, color_theme="plasma")
    wc.draw_image(save_file=True, image_name="01_stopwords_and_min_word_length")

    print("Saved Results/01_stopwords_and_min_word_length.png")
    print("Tip: try editing stopwords to include 'The' and re-run to see the difference.")


if __name__ == "__main__":
    main()


