import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union

from .logging_config import get_logger

logger = get_logger(__name__)


class MaskError(Exception):
    """Exception raised for mask-related errors."""
    pass


class MaskProcessor:
    """
    Process image masks to support shaped wordclouds.

    Mask uses 1 for available pixels, 0 for blocked pixels.
    """

    def __init__(self, mask_image: Union[str, Path, Image.Image, np.ndarray], threshold: int = 200):
        logger.info(f"Initializing mask processor with threshold={threshold}")
        self.mask = self._load_mask(mask_image, threshold)
        self.height, self.width = self.mask.shape
        logger.info(f"Mask dimensions: {self.width}x{self.height}")
        self._initialize_integral()
        available_pixels = np.sum(self.mask == 1)
        total_pixels = self.height * self.width
        logger.debug(f"Available pixels: {available_pixels}/{total_pixels} ({100*available_pixels/total_pixels:.1f}%)")

    def _load_mask(self, mask_image: Union[str, Path, Image.Image, np.ndarray], threshold: int) -> np.ndarray:
        if isinstance(mask_image, np.ndarray):
            logger.debug("Loading mask from numpy array")
            if mask_image.ndim == 3:
                mask = np.array(Image.fromarray(mask_image).convert("L"))
            elif mask_image.ndim == 2:
                mask = mask_image
            else:
                logger.error(f"Unsupported numpy array dimensions: {mask_image.ndim}")
                raise TypeError(f"Unsupported numpy array dimensions: {mask_image.ndim}")
        elif isinstance(mask_image, (str, Path)):
            logger.debug(f"Loading mask image from file: {mask_image}")
            try:
                mask = np.array(Image.open(mask_image).convert("L"))
                logger.info(f"Successfully loaded mask image from {mask_image}")
            except FileNotFoundError:
                logger.error(f"Mask image file not found: {mask_image}")
                raise MaskError(f"Mask image file not found: {mask_image}")
            except Exception as e:
                logger.error(f"Error loading mask image {mask_image}: {e}")
                raise MaskError(f"Error loading mask image: {e}")
        elif isinstance(mask_image, Image.Image):
            logger.debug("Loading mask from PIL Image")
            mask = np.array(mask_image.convert("L"))
        else:
            logger.error("Invalid mask type provided")
            raise TypeError("Mask must be a file path, PIL Image, or numpy array")

        if mask.size == 0:
            logger.error("Mask image is empty")
            raise MaskError("Mask image is empty.")

        # If already binary (0/1), keep as-is; otherwise apply threshold
        if mask.max() <= 1:
            logger.debug("Mask is already binary")
            return mask.astype(np.uint8)
        logger.debug(f"Binarizing mask with threshold {threshold}")
        return (mask > threshold).astype(np.uint8)

    def _initialize_integral(self) -> None:
        logger.debug("Initializing integral image for mask")
        inverted_mask = 1 - self.mask
        self.integral = np.zeros((self.height + 1, self.width + 1), dtype=np.uint64)
        self.integral[1:, 1:] = np.cumsum(np.cumsum(inverted_mask, axis=0), axis=1)
        logger.debug("Integral image initialized")

    def is_position_available(self, pos_y: int, pos_x: int, size_y: int, size_x: int) -> bool:
        if (
            pos_x < 0
            or pos_y < 0
            or pos_y + size_y > self.height
            or pos_x + size_x > self.width
        ):
            return False

        # Cast to Python ints to avoid uint64 overflow warnings during subtraction.
        sum_unavailable = (
            int(self.integral[pos_y + size_y, pos_x + size_x])
            - int(self.integral[pos_y, pos_x + size_x])
            - int(self.integral[pos_y + size_y, pos_x])
            + int(self.integral[pos_y, pos_x])
        )
        return sum_unavailable == 0

