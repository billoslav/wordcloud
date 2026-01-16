"""
Tests for SVG fidelity against PIL font metrics.
"""

from __future__ import annotations

import re

from PIL import ImageFont

from wordcloud.utils.export import SVGExporter


def test_svg_baseline_alignment_and_length() -> None:
    font_path = "fonts/Arial Unicode.ttf"
    word = "Kerning"
    font_size = 32
    position = (10, 20)

    gen_positions = [
        ((word, 1.0, 1), font_path, font_size, position, None, "#000000"),
    ]

    svg = SVGExporter().generate_svg_content(
        gen_positions=gen_positions,
        width=200,
        height=100,
        default_font_path=font_path,
        max_font_size=font_size,
        background_color="white",
    )

    match = re.search(r'transform="translate\(([-\d.]+),([-\d.]+)\)"', svg)
    assert match is not None
    svg_x = float(match.group(1))
    svg_y = float(match.group(2))

    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(word)
    ascent, _ = font.getmetrics()
    expected_x = position[0] - bbox[0]
    expected_y = position[1] + ascent

    assert abs(svg_x - expected_x) < 1.0
    assert abs(svg_y - expected_y) < 1.0

    length_match = re.search(r'textLength="([-\d.]+)"', svg)
    assert length_match is not None
    text_length = float(length_match.group(1))

    expected_length = font.getlength(word) if hasattr(font, "getlength") else bbox[2] - bbox[0]
    assert abs(text_length - expected_length) < 1.0
