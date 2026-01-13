from .integral_image import IntegralImage, StaticIntegralImage, STRATEGIES
from .text_processing import TextProcessor
from .placement import find_position_random, find_position_rectangular_spiral
from .trace_utils import Tracer
from .font_utils import FontCache, FontError
from .mask import MaskProcessor, MaskError
from .visualization import (
    COLOR_THEMES,
    generate_color_gradient,
    generate_colors_by_frequency,
    generate_colors_by_sentiment,
    generate_colors_by_length,
    hex_to_rgb,
    rgb_to_hex,
)
from .logging_config import setup_logging, get_logger
from .performance import PerformanceTracker, Timer, Profiler
from .collision import (
    CollisionDetector,
    BruteForceCollisionDetector,
    GridCollisionSystem,
    QuadtreeCollisionDetector,
    MaskCollisionDetector,
    create_collision_detector,
    rectangles_overlap,
    point_in_rectangle,
)
from .config import ConfigManager, get_config, DEFAULT_CONFIG
from .presets import get_preset, list_presets, PresetManager, create_wordcloud_from_preset

__all__ = [
    'IntegralImage',
    'StaticIntegralImage',
    'STRATEGIES',
    'TextProcessor',
    'find_position_random',
    'find_position_rectangular_spiral',
    'Tracer',
    'FontCache',
    'FontError',
    'MaskProcessor',
    'MaskError',
    'COLOR_THEMES',
    'generate_color_gradient',
    'generate_colors_by_frequency',
    'generate_colors_by_sentiment',
    'generate_colors_by_length',
    'hex_to_rgb',
    'rgb_to_hex',
    'setup_logging',
    'get_logger',
    'PerformanceTracker',
    'Timer',
    'Profiler',
    # Collision detection
    'CollisionDetector',
    'BruteForceCollisionDetector',
    'GridCollisionSystem',
    'QuadtreeCollisionDetector',
    'MaskCollisionDetector',
    'create_collision_detector',
    'rectangles_overlap',
    'point_in_rectangle',
    # Configuration
    'ConfigManager',
    'get_config',
    'DEFAULT_CONFIG',
    # Presets
    'get_preset',
    'list_presets',
    'PresetManager',
    'create_wordcloud_from_preset',
]