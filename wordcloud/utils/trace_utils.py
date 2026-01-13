"""
Tracing helpers to visualize placement attempts.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

from PIL import Image, ImageDraw
from .helpers import create_tracking_structure

TRACE_MARGIN = 200


def save_trace_img(trace_img: Image.Image, trace_img_name: Path) -> None:
    """Persist the trace image if set."""
    if trace_img and trace_img_name:
        trace_img.save(trace_img_name)


def draw_trace_point(draw: ImageDraw.ImageDraw, pos_x: int, pos_y: int, color: str = "red") -> None:
    """Draw a small point offset by the margin."""
    cx = pos_x + TRACE_MARGIN // 2
    cy = pos_y + TRACE_MARGIN // 2
    draw.point([(cx, cy)], fill=color)

class Tracer:
    """
    Simple tracer that captures placement attempts to a PNG for debugging.
    """

    def __init__(self) -> None:
        self.is_active = False
        self.trace_img: Optional[Image.Image] = None
        self.trace_draw: Optional[ImageDraw.ImageDraw] = None
        self.trace_img_name: Optional[Path] = None

    def setup(
        self,
        canvas_width: int,
        canvas_height: int,
        base_tracking_dir: Path,
        place_strategy: str,
        word_info: tuple[str, float, int],
        word_size: tuple[int, int],
        mask_array=None,
    ) -> None:
        strategy_dir = create_tracking_structure(base_tracking_dir, place_strategy)

        # Create base tracing image
        self.trace_img = Image.new(
            "RGB", (canvas_width + TRACE_MARGIN, canvas_height + TRACE_MARGIN), color="white"
        )
        self.trace_draw = ImageDraw.Draw(self.trace_img)
        self.is_active = True

        # Draw mask if provided (1 = available)
        if mask_array is not None:
            mask_img = Image.fromarray(mask_array.astype("uint8") * 255, mode="L")
            self.trace_img.paste(mask_img.convert("RGB"), (TRACE_MARGIN // 2, TRACE_MARGIN // 2))

        # Draw canvas boundary rectangle and word size indicators
        size_x, size_y = word_size
        self.trace_draw.rectangle([(TRACE_MARGIN // 2, TRACE_MARGIN // 2), ((canvas_width + (TRACE_MARGIN // 2), canvas_height + (TRACE_MARGIN // 2)))], fill=None, outline="grey", width=1)
        self.trace_draw.line([(canvas_width + TRACE_MARGIN // 2 - size_x, 0), (canvas_width + TRACE_MARGIN // 2 - size_x, canvas_height + TRACE_MARGIN)], fill="red", width=1, joint=None)
        self.trace_draw.line([(0, canvas_height + TRACE_MARGIN // 2 - size_y), (canvas_width + TRACE_MARGIN, canvas_height + TRACE_MARGIN // 2 - size_y)], fill="red", width=1, joint=None)

        # Name trace image; include word text for readability
        word_text = word_info[0] if word_info else "word"
        self.trace_img_name = str(strategy_dir / f"trace_{word_text}_{time.time()}.png")

    def draw_point(self, pos_x: int, pos_y: int, color: str = "red") -> None:
        if not self.is_active or not self.trace_draw:
            return
        draw_trace_point(self.trace_draw, pos_x, pos_y, color=color)

    def save(self) -> None:
        if not self.is_active or not self.trace_img or not self.trace_img_name:
            return
        save_trace_img(self.trace_img, self.trace_img_name)
        self.is_active = False


def tracing_setup(
    canvas_width: int,
    canvas_height: int,
    base_tracking_dir: Path,
    place_strategy: str,
    word_info: tuple[str, float, int],
    word_size: tuple[int, int],
    mask_array=None,
) -> Tracer:
    """
    Convenience wrapper to create and configure a tracer.
    """
    tracer = Tracer()
    tracer.setup(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        base_tracking_dir=base_tracking_dir,
        place_strategy=place_strategy,
        word_info=word_info,
        word_size=word_size,
        mask_array=mask_array,
    )
    return tracer

