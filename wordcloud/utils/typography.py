"""
Advanced typography and visual effects utilities for wordcloud generation.

This module provides text effects (outline, shadow, gradients), advanced font
size distribution, and per-word font selection capabilities.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Optional, Dict, List, Tuple, Union, Callable, Any

from .logging_config import get_logger

logger = get_logger(__name__)


def apply_logarithmic_scaling(frequency: float, min_freq: float = 0.0, max_freq: float = 1.0) -> float:
    """
    Apply logarithmic scaling to a frequency value.
    
    Args:
        frequency: Normalized frequency (0.0 to 1.0)
        min_freq: Minimum frequency value (default: 0.0)
        max_freq: Maximum frequency value (default: 1.0)
        
    Returns:
        Logarithmically scaled frequency value
    """
    if frequency <= 0:
        return 0.0
    
    # Normalize to [0, 1] range
    normalized = (frequency - min_freq) / (max_freq - min_freq) if max_freq > min_freq else 0.0
    normalized = max(0.0, min(1.0, normalized))
    
    # Apply logarithmic scaling: log(x + 1) / log(2)
    scaled = math.log(normalized + 1) / math.log(2)
    return scaled


def apply_power_law_scaling(frequency: float, exponent: float = 0.5, 
                            min_freq: float = 0.0, max_freq: float = 1.0) -> float:
    """
    Apply power-law scaling to a frequency value.
    
    Args:
        frequency: Normalized frequency (0.0 to 1.0)
        exponent: Power-law exponent (default: 0.5 for square root)
        min_freq: Minimum frequency value (default: 0.0)
        max_freq: Maximum frequency value (default: 1.0)
        
    Returns:
        Power-law scaled frequency value
    """
    if frequency <= 0:
        return 0.0
    
    # Normalize to [0, 1] range
    normalized = (frequency - min_freq) / (max_freq - min_freq) if max_freq > min_freq else 0.0
    normalized = max(0.0, min(1.0, normalized))
    
    # Apply power-law scaling: x^exponent
    scaled = math.pow(normalized, exponent)
    return scaled


def calculate_font_size(frequency: float, min_font_size: int, max_font_size: int,
                       distribution: str = 'linear', 
                       distribution_params: Optional[Dict[str, Any]] = None,
                       custom_scaling: Optional[Callable[[float], float]] = None) -> int:
    """
    Calculate font size based on frequency and distribution method.
    
    Args:
        frequency: Normalized frequency (0.0 to 1.0)
        min_font_size: Minimum font size in points
        max_font_size: Maximum font size in points
        distribution: Distribution method ('linear', 'logarithmic', 'power', 'custom')
        distribution_params: Optional parameters for distribution (e.g., {'exponent': 0.5})
        custom_scaling: Custom scaling function (used when distribution='custom')
        
    Returns:
        Calculated font size in points
    """
    if frequency <= 0:
        return min_font_size
    
    distribution_params = distribution_params or {}
    
    # Apply scaling based on distribution method
    if distribution == 'logarithmic':
        scaled_freq = apply_logarithmic_scaling(frequency)
    elif distribution == 'power':
        exponent = distribution_params.get('exponent', 0.5)
        scaled_freq = apply_power_law_scaling(frequency, exponent=exponent)
    elif distribution == 'custom' and custom_scaling:
        scaled_freq = custom_scaling(frequency)
    else:  # 'linear' or default
        scaled_freq = frequency
    
    # Map scaled frequency to font size range
    font_size = min_font_size + (max_font_size - min_font_size) * scaled_freq
    return int(round(font_size))


def assign_fonts_to_words(words: List[Tuple[str, float, int]], 
                          font_paths: Union[str, List[str], Dict[str, str]],
                          strategy: str = 'frequency',
                          rng: Optional[random.Random] = None) -> Dict[str, str]:
    """
    Assign fonts to words based on a strategy.
    
    Args:
        words: List of (word, normalized_freq, original_freq) tuples
        font_paths: Single font path, list of font paths, or dict mapping words to fonts
        strategy: Assignment strategy ('frequency', 'random', 'category')
        
    Returns:
        Dictionary mapping words to font paths
    """
    word_fonts = {}
    
    # If single font, assign to all words
    if isinstance(font_paths, str):
        return {word: font_paths for word, _, _ in words}
    
    # If dict, use direct mapping
    if isinstance(font_paths, dict):
        return {word: font_paths.get(word, font_paths.get(list(font_paths.keys())[0] if font_paths else '')) 
                for word, _, _ in words}
    
    # If list, apply strategy
    if not isinstance(font_paths, list) or not font_paths:
        logger.warning("Invalid font_paths format, using first font")
        return {word: font_paths[0] if isinstance(font_paths, list) and font_paths else '' 
                for word, _, _ in words}
    
    num_fonts = len(font_paths)
    
    if strategy == 'frequency':
        # Assign fonts based on frequency (high frequency = first fonts)
        sorted_words = sorted(words, key=lambda x: x[1], reverse=True)
        for i, (word, freq, _) in enumerate(sorted_words):
            font_idx = min(i * num_fonts // len(words), num_fonts - 1)
            word_fonts[word] = font_paths[font_idx]
    
    elif strategy == 'random':
        # Random assignment
        rng = rng or random
        for word, _, _ in words:
            word_fonts[word] = rng.choice(font_paths)
    
    elif strategy == 'category':
        # Category-based (could be extended with word categories)
        # For now, use frequency-based as fallback
        sorted_words = sorted(words, key=lambda x: x[1], reverse=True)
        for i, (word, freq, _) in enumerate(sorted_words):
            font_idx = min(i * num_fonts // len(words), num_fonts - 1)
            word_fonts[word] = font_paths[font_idx]
    
    else:
        logger.warning(f"Unknown font assignment strategy '{strategy}', using first font")
        for word, _, _ in words:
            word_fonts[word] = font_paths[0]
    
    return word_fonts


class TextEffects:
    """
    Container for text effect configuration.
    """
    def __init__(self, 
                 outline: Optional[Dict[str, Any]] = None,
                 shadow: Optional[Dict[str, Any]] = None,
                 gradient: Optional[Dict[str, Any]] = None):
        """
        Initialize text effects.
        
        Args:
            outline: Outline configuration dict with 'width' and 'color' keys
            shadow: Shadow configuration dict with 'offset_x', 'offset_y', 'blur', 'color', 'opacity' keys
            gradient: Gradient configuration dict with 'start_color', 'end_color' keys
        """
        self.outline = outline or {}
        self.shadow = shadow or {}
        self.gradient = gradient or {}
    
    def has_outline(self) -> bool:
        """Check if outline effect is enabled."""
        return bool(self.outline and self.outline.get('width', 0) > 0)
    
    def has_shadow(self) -> bool:
        """Check if shadow effect is enabled."""
        return bool(self.shadow and (self.shadow.get('offset_x', 0) != 0 or 
                                     self.shadow.get('offset_y', 0) != 0))
    
    def has_gradient(self) -> bool:
        """Check if gradient effect is enabled."""
        return bool(self.gradient and self.gradient.get('start_color') and 
                   self.gradient.get('end_color'))


def validate_rotation_angles(angles: Tuple[int, ...]) -> bool:
    """
    Validate rotation angles (now supports any angle, not just 90/-90).
    
    Args:
        angles: Tuple of rotation angles in degrees
        
    Returns:
        True if valid
    """
    if not isinstance(angles, tuple) or not angles:
        return False
    
    for angle in angles:
        if not isinstance(angle, int) or angle < -360 or angle > 360:
            return False
    
    return True


def normalize_rotation_angle(angle: int) -> int:
    """
    Normalize rotation angle to [-360, 360] range.
    
    Args:
        angle: Rotation angle in degrees
        
    Returns:
        Normalized angle
    """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle

