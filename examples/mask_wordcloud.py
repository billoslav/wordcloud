import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud import Wordcloud

from _text_sources import load_example_text


def main():
    radius = 150
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = (x * x + y * y <= radius * radius).astype(np.uint8)

    wc = Wordcloud(width=2 * radius, height=2 * radius, mask_image=mask, background_color="white")
    wc.generate(load_example_text("generated.txt")).draw_image(save_file=True, image_name="example_masked")


if __name__ == "__main__":
    main()

