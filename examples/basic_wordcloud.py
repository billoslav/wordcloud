import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud

from _text_sources import load_example_text


def main():
    text = load_example_text("generated.txt")
    wc = Wordcloud(width=600, height=338)
    wc.generate(text)
    wc.draw_image(save_file=True, image_name="example_basic")


if __name__ == "__main__":
    main()

