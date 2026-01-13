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

    wc = Wordcloud(width=800, height=400, font_path=str(font_path), background_color="white")
    wc.generate(load_example_text("lorem.txt"), color_theme="viridis")

    wc.generate_svg(save_file=True, file_name="05_wordcloud")
    wc.create_html(save_file=True, file_name="05_wordcloud", interactive=True, standalone=True)

    print("Saved Results/05_wordcloud.svg")
    print("Saved Results/05_wordcloud.html")


if __name__ == "__main__":
    main()


