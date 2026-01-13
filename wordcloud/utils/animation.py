"""
Animation utilities for wordcloud generation.

This module provides frame-by-frame generation for animated wordclouds,
supporting various animation styles and timing controls.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum

from .logging_config import get_logger

logger = get_logger(__name__)


class AnimationStyle(Enum):
    """Animation styles for word appearance."""
    FADE_IN = "fade_in"
    SLIDE_IN = "slide_in"
    SCALE_IN = "scale_in"
    RANDOM = "random"


class AnimationFrame:
    """
    Represents a single frame in an animation sequence.
    """
    def __init__(self, frame_index: int, words: List[Tuple], 
                 opacity: float = 1.0, 
                 transform: Optional[Dict[str, Any]] = None):
        """
        Initialize an animation frame.
        
        Args:
            frame_index: Index of this frame in the sequence
            words: List of word tuples (word_data, font_path, font_size, position, orientation, color)
            opacity: Opacity of words in this frame (0.0 to 1.0)
            transform: Optional transform dict with 'translate', 'scale', 'rotate' keys
        """
        self.frame_index = frame_index
        self.words = words
        self.opacity = opacity
        self.transform = transform or {}


def generate_animation_frames(
    all_words: List[Tuple],
    style: AnimationStyle = AnimationStyle.FADE_IN,
    frames_per_word: int = 3,
    total_duration_ms: Optional[int] = None,
    per_word_delay_ms: int = 100,
    per_word_duration_ms: int = 300
) -> List[AnimationFrame]:
    """
    Generate animation frames from a list of words.
    
    Args:
        all_words: List of all word tuples to animate
        style: Animation style
        frames_per_word: Number of frames per word appearance
        total_duration_ms: Total animation duration in milliseconds (optional)
        per_word_delay_ms: Delay before each word appears (milliseconds)
        per_word_duration_ms: Duration for each word animation (milliseconds)
        
    Returns:
        List of AnimationFrame objects
    """
    frames = []
    num_words = len(all_words)
    
    if num_words == 0:
        return frames
    
    # Calculate timing
    if total_duration_ms:
        per_word_duration_ms = total_duration_ms // num_words
        per_word_delay_ms = 0
    
    current_words = []
    
    for word_idx, word in enumerate(all_words):
        # Calculate frame range for this word
        start_frame = word_idx * (frames_per_word + per_word_delay_ms // (per_word_duration_ms // frames_per_word))
        
        for frame_offset in range(frames_per_word):
            frame_idx = start_frame + frame_offset
            
            # Add word to current words list
            if frame_offset == 0:
                current_words.append(word)
            
            # Calculate animation parameters based on style
            progress = frame_offset / frames_per_word
            
            if style == AnimationStyle.FADE_IN:
                opacity = progress
                transform = {}
            elif style == AnimationStyle.SLIDE_IN:
                opacity = 1.0
                slide_distance = 50 * (1 - progress)
                transform = {'translate': (slide_distance, 0)}
            elif style == AnimationStyle.SCALE_IN:
                opacity = 1.0
                scale = 0.1 + 0.9 * progress
                transform = {'scale': scale}
            elif style == AnimationStyle.RANDOM:
                # Random style: randomly choose fade, slide, or scale
                import random
                rand_style = random.choice([AnimationStyle.FADE_IN, AnimationStyle.SLIDE_IN, AnimationStyle.SCALE_IN])
                if rand_style == AnimationStyle.FADE_IN:
                    opacity = progress
                    transform = {}
                elif rand_style == AnimationStyle.SLIDE_IN:
                    opacity = 1.0
                    slide_distance = 50 * (1 - progress)
                    transform = {'translate': (random.randint(-50, 50), random.randint(-50, 50))}
                else:
                    opacity = 1.0
                    scale = 0.1 + 0.9 * progress
                    transform = {'scale': scale}
            else:
                opacity = 1.0
                transform = {}
            
            # Create frame with current words
            frame = AnimationFrame(
                frame_index=frame_idx,
                words=current_words.copy(),
                opacity=opacity if word_idx == len(current_words) - 1 else 1.0,
                transform=transform if word_idx == len(current_words) - 1 else {}
            )
            frames.append(frame)
    
    # Ensure we have at least one frame with all words
    if frames:
        final_frame = AnimationFrame(
            frame_index=frames[-1].frame_index + 1,
            words=all_words,
            opacity=1.0,
            transform={}
        )
        frames.append(final_frame)
    
    return frames


def apply_animation_to_word(word_data: Tuple, frame: AnimationFrame, word_index: int) -> Tuple:
    """
    Apply animation transform to a single word.
    
    Args:
        word_data: Word tuple (word_data, font_path, font_size, position, orientation, color)
        frame: Animation frame
        word_index: Index of word in frame.words
        
    Returns:
        Modified word tuple with animation applied
    """
    if word_index >= len(frame.words):
        return word_data
    
    word_data_tuple, font_path, font_size, position, orientation, color = word_data
    
    # Apply transform if this is the last word (being animated)
    if word_index == len(frame.words) - 1 and frame.transform:
        x, y = position
        
        if 'translate' in frame.transform:
            tx, ty = frame.transform['translate']
            x += tx
            y += ty
        
        if 'scale' in frame.transform:
            scale = frame.transform['scale']
            font_size = int(font_size * scale)
        
        position = (x, y)
    
    return (word_data_tuple, font_path, font_size, position, orientation, color)

