import numpy as np
import pytest

from wordcloud import Wordcloud


def test_mask_blocks_center():
    mask = np.ones((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 0  # blocked center

    wc = Wordcloud(
        width=20,
        height=20,
        mask_image=mask,
        min_word_length=1,
        min_font_size=1,
        max_font_size=1,
        font_step=1,
        margin=0,
        place_strategy="brute",
        stopwords=[],
    )
    wc.generate("a")

    assert wc.gen_positions, "No positions generated with mask applied"
    pos = wc.gen_positions[0][3]  # (x, y)
    # Ensure placed position is within available mask area
    assert wc.mask_processor.is_position_available(pos[1], pos[0], 1, 1)


def test_mask_dimension_mismatch_raises():
    mask = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        Wordcloud(width=20, height=20, mask_image=mask)

