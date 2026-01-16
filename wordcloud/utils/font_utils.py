from PIL import Image, ImageFont, ImageDraw
from typing import Dict, Tuple, Optional, Any
import logging

from .logging_config import get_logger

logger = get_logger(__name__)


class FontError(Exception):
    """Exception raised for font-related errors."""
    pass


class FontCache:
    """
    Cache for fonts and text bounding boxes to avoid repeated metric lookups.
    """

    def __init__(self):
        self._font_cache: Dict[Tuple[str, int], Any] = {}
        self._bbox_cache: Dict[Tuple[str, str, int, Optional[int]], Tuple[int, int, int, int]] = {}
        self._rotated_bbox_cache: Dict[Tuple[str, str, int, int], Tuple[int, int, int, int]] = {}
        logger.info("FontCache initialized")

    def get_font(self, font_path: str, font_size: int) -> Any:
        if not isinstance(font_path, str) or not font_path:
            raise TypeError("font_path must be a non-empty string")
        if not isinstance(font_size, int) or font_size <= 0:
            raise TypeError("font_size must be a positive integer")

        key = (font_path, font_size)
        if key not in self._font_cache:
            logger.debug(f"Cache miss for font: {key}. Loading font.")
            try:
                font = ImageFont.truetype(font_path, font_size)
                self._font_cache[key] = font
            except IOError as exc:
                logger.error(f"Could not load font file '{font_path}': {exc}")
                raise FontError(f"Could not load font file '{font_path}': {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Unexpected error loading font '{font_path}': {exc}")
                raise FontError(f"Unexpected error loading font '{font_path}': {exc}")
        else:
            logger.debug(f"Cache hit for font: {key}")
        return self._font_cache[key]

    def get_text_bbox(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font_path: str,
        font_size: int,
        orientation: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        key = (text, font_path, font_size, orientation)
        if key not in self._bbox_cache:
            logger.debug(f"Cache miss for bbox: {key}. Calculating.")
            font = self.get_font(font_path, font_size)
            if orientation is not None and orientation != 0:
                font = ImageFont.TransposedFont(font, orientation=orientation)
            bbox = draw.textbbox((0, 0), text, font=font, anchor="lt")
            self._bbox_cache[key] = bbox
        else:
            logger.debug(f"Cache hit for bbox: {key}")
        return self._bbox_cache[key]

    def get_rotated_bbox(
        self,
        text: str,
        font_path: str,
        font_size: int,
        rotation_degrees: int,
    ) -> Tuple[int, int, int, int]:
        """
        Return a bounding box for text rotated by an arbitrary angle.
        """
        if not isinstance(rotation_degrees, int):
            raise TypeError("rotation_degrees must be an integer")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        key = (text, font_path, font_size, rotation_degrees)
        if key in self._rotated_bbox_cache:
            return self._rotated_bbox_cache[key]

        font = self.get_font(font_path, font_size)
        bbox = font.getbbox(text)
        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        if width == 0 or height == 0:
            rotated_bbox = (0, 0, 0, 0)
            self._rotated_bbox_cache[key] = rotated_bbox
            return rotated_bbox

        base = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(base)
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill=255)
        rotated = base.rotate(rotation_degrees, expand=True, resample=Image.BICUBIC)
        rotated_bbox = (0, 0, rotated.size[0], rotated.size[1])
        self._rotated_bbox_cache[key] = rotated_bbox
        return rotated_bbox

    def clear(self) -> None:
        """Clear all cached font and bbox entries."""
        self._font_cache.clear()
        self._bbox_cache.clear()
        self._rotated_bbox_cache.clear()
        logger.debug("FontCache cleared")
