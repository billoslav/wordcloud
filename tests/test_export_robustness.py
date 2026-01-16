"""
Tests for export validation and dependency handling.
"""

from __future__ import annotations

import pytest # type: ignore

from PIL import Image

import wordcloud.utils.export as export_utils
from wordcloud.utils.export import AnimatedGIFExporter, PDFExporter, SVGExporter, export_image


def test_export_image_rejects_directory(tmp_path) -> None:
    image = Image.new("RGB", (10, 10), "white")
    with pytest.raises(ValueError, match="output path must be a file"):
        export_image(image, tmp_path)


def test_pdf_export_requires_reportlab(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(export_utils, "REPORTLAB_AVAILABLE", False)
    image = Image.new("RGB", (10, 10), "white")
    with pytest.raises(RuntimeError, match="ReportLab is required"):
        PDFExporter().export_pdf(image, tmp_path / "out.pdf")


def test_gif_export_requires_pil(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(export_utils, "PIL_AVAILABLE", False)
    frames = [Image.new("RGB", (10, 10), "white")]
    with pytest.raises(RuntimeError, match="PIL is required"):
        AnimatedGIFExporter().export_animated_gif(frames, tmp_path / "out.gif")


def test_svg_export_invalid_font_fallback() -> None:
    exporter = SVGExporter()
    gen_positions = [
        (("test", 1.0, 1), "/no/such/font.ttf", 20, (10, 10), None, "#000000"),
    ]
    svg_content = exporter.generate_svg_content(
        gen_positions=gen_positions,
        width=200,
        height=100,
        default_font_path="/no/such/font.ttf",
        max_font_size=20,
        background_color="white",
    )
    assert "<svg" in svg_content
