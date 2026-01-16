"""
Image-based color extraction utilities for wordcloud generation.

This module provides functionality to extract colors from images and
use them for wordcloud coloring.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional
from collections import Counter

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    import numpy as np
    from PIL import Image
    NUMPY_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    PIL_AVAILABLE = False
    logger.debug("NumPy or PIL not available, color extraction disabled")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.debug("scikit-learn not available, advanced color extraction disabled")


def extract_colors_from_image(
    image_path: str,
    num_colors: int = 10,
    method: str = 'kmeans',
    resize_to: Optional[Tuple[int, int]] = None,
    rng: Optional["np.random.Generator"] = None,
) -> List[str]:
    """
    Extract dominant colors from an image.
    
    Args:
        image_path: Path to the image file
        num_colors: Number of colors to extract
        method: Extraction method ('kmeans', 'most_common', 'random')
        resize_to: Optional tuple (width, height) to resize image before processing
        
    Returns:
        List of hex color strings
    """
    if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
        raise RuntimeError("PIL and NumPy are required for color extraction")
    
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if requested (for performance)
        if resize_to:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        
        if method == 'kmeans' and SKLEARN_AVAILABLE:
            # Use K-means clustering for better color extraction
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
        elif method == 'most_common':
            # Get most common colors
            pixel_tuples = [tuple(pixel) for pixel in pixels]
            color_counts = Counter(pixel_tuples)
            most_common = color_counts.most_common(num_colors)
            colors = np.array([list(color) for color, _ in most_common])
        else:  # 'random' or fallback
            # Random sampling
            rng = rng or np.random.default_rng()
            indices = rng.choice(len(pixels), size=num_colors, replace=False)
            colors = pixels[indices]
        
        # Convert to hex strings
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        
        logger.debug(f"Extracted {len(hex_colors)} colors from image: {image_path}")
        return hex_colors
        
    except Exception as e:
        logger.error(f"Failed to extract colors from image {image_path}: {e}")
        raise


def extract_colors_from_pil_image(
    image: Image.Image,
    num_colors: int = 10,
    method: str = 'kmeans',
    rng: Optional["np.random.Generator"] = None,
) -> List[str]:
    """
    Extract dominant colors from a PIL Image object.
    
    Args:
        image: PIL Image object
        num_colors: Number of colors to extract
        method: Extraction method ('kmeans', 'most_common', 'random')
        
    Returns:
        List of hex color strings
    """
    if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
        raise RuntimeError("PIL and NumPy are required for color extraction")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    
    if method == 'kmeans' and SKLEARN_AVAILABLE:
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
    elif method == 'most_common':
        pixel_tuples = [tuple(pixel) for pixel in pixels]
        color_counts = Counter(pixel_tuples)
        most_common = color_counts.most_common(num_colors)
        colors = np.array([list(color) for color, _ in most_common])
    else:  # 'random' or fallback
        rng = rng or np.random.default_rng()
        indices = rng.choice(len(pixels), size=num_colors, replace=False)
        colors = pixels[indices]
    
    # Convert to hex strings
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
    
    logger.debug(f"Extracted {len(hex_colors)} colors from PIL image")
    return hex_colors


def map_words_to_image_colors(
    words: List[str],
    image_path: str,
    num_colors: int = 10,
    method: str = 'kmeans',
    frequency_based: bool = True,
    rng: Optional["np.random.Generator"] = None,
) -> Dict[str, str]:
    """
    Map words to colors extracted from an image.
    
    Args:
        words: List of words to color
        image_path: Path to the image file
        num_colors: Number of colors to extract
        method: Extraction method
        frequency_based: If True, assign colors based on word frequency order
        
    Returns:
        Dictionary mapping words to hex color strings
    """
    colors = extract_colors_from_image(image_path, num_colors, method, rng=rng)
    
    word_colors = {}
    if frequency_based:
        # Assign colors based on word order (assuming words are sorted by frequency)
        for i, word in enumerate(words):
            color_idx = min(i * len(colors) // len(words), len(colors) - 1)
            word_colors[word] = colors[color_idx]
    else:
        # Random assignment
        import random
        if rng is not None:
            rng_indices = rng.choice(len(colors), size=len(words), replace=True)
            for word, idx in zip(words, rng_indices):
                word_colors[word] = colors[int(idx)]
        else:
            for word in words:
                word_colors[word] = random.choice(colors)
    
    return word_colors

