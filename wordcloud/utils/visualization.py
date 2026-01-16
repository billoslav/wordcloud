"""
Visualization helpers: color themes, gradients, and sentiment-based coloring.
"""

from __future__ import annotations

import colorsys
import logging
import random
from typing import Dict, List, Tuple, Union, Optional


logger = logging.getLogger(__name__)

COLOR_THEMES = {
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "pastel": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd"],
    "bright": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"],
    "dark": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    "viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
    "magma": ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55964", "#fb8761", "#fec287", "#fbfdbf"],
    "inferno": ["#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746"],
    "plasma": ["#0d0887", "#41049d", "#6a00a8", "#8f0da4", "#b12a90", "#d04a7e", "#e87364", "#f89441", "#fdc328", "#f0f921"],
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def generate_color_gradient(
    start_color: Union[Tuple[int, int, int], str],
    end_color: Union[Tuple[int, int, int], str],
    num_colors: int,
) -> List[str]:
    if isinstance(start_color, str):
        start_color = hex_to_rgb(start_color)
    if isinstance(end_color, str):
        end_color = hex_to_rgb(end_color)

    colors: List[str] = []
    for i in range(num_colors):
        t = i / (num_colors - 1) if num_colors > 1 else 0
        r = int(start_color[0] * (1 - t) + end_color[0] * t)
        g = int(start_color[1] * (1 - t) + end_color[1] * t)
        b = int(start_color[2] * (1 - t) + end_color[2] * t)
        colors.append(rgb_to_hex((r, g, b)))
    return colors


def generate_colors_by_frequency(
    frequencies: Dict[str, float],
    color_theme: str = "viridis",
    random_colors: bool = False,
    shuffle: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, str]:
    colors = COLOR_THEMES.get(color_theme, COLOR_THEMES["viridis"]).copy()
    rng = rng or random
    if shuffle:
        rng.shuffle(colors)
    sorted_words = sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True)
    word_colors: Dict[str, str] = {}
    if random_colors:
        for word in sorted_words:
            word_colors[word] = rng.choice(colors)
    else:
        num_words = len(sorted_words) or 1
        for i, word in enumerate(sorted_words):
            idx = min(int(i * len(colors) / num_words), len(colors) - 1)
            word_colors[word] = colors[idx]
    return word_colors


def generate_colors_by_sentiment(
    words: List[str],
    sentiments: Dict[str, float],
    neutral_color: str = "#808080",
) -> Dict[str, str]:
    positive_color = "#4CAF50"
    negative_color = "#F44336"
    word_colors: Dict[str, str] = {}
    for word in words:
        if word not in sentiments:
            word_colors[word] = neutral_color
            continue
        sentiment = sentiments[word]
        if sentiment > 0.05:
            intensity = min(1.0, sentiment * 2)
            r, g, b = hex_to_rgb(positive_color)
            r = int(r * (1 - intensity * 0.7))
            b = int(b * (1 - intensity * 0.7))
            word_colors[word] = rgb_to_hex((r, g, b))
        elif sentiment < -0.05:
            intensity = min(1.0, abs(sentiment) * 2)
            r, g, b = hex_to_rgb(negative_color)
            g = int(g * (1 - intensity * 0.7))
            b = int(b * (1 - intensity * 0.7))
            word_colors[word] = rgb_to_hex((r, g, b))
        else:
            word_colors[word] = neutral_color
    return word_colors


def generate_colors_by_length(words: List[str], min_length: int = 3, max_length: int = 12) -> Dict[str, str]:
    word_colors: Dict[str, str] = {}
    for word in words:
        length = len(word)
        normalized = min(max((length - min_length) / max(1, max_length - min_length), 0), 1)
        hue = (1 - normalized) * 0.6  # 0.0=red to 0.6=blue/green
        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        word_colors[word] = rgb_to_hex(rgb)
    return word_colors

