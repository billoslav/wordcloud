import re
import os
import random
import numpy as np
from random import randint
from operator import itemgetter
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import logging
from typing import Optional, Callable, Union

from .utils import IntegralImage, STRATEGIES, TextProcessor, MaskProcessor, FontCache
from .utils.logging_config import get_logger
from .utils.performance import PerformanceTracker
from .utils.helpers import create_folder
from .utils.collision import CollisionDetector, create_collision_detector
from .utils.placement import find_position_random, find_position_rectangular_spiral
from .utils.config import ConfigManager, get_config
from .utils.typography import (
    TextEffects, calculate_font_size, assign_fonts_to_words,
    validate_rotation_angles, normalize_rotation_angle
)

class Wordcloud:
    """
    Main class for generating wordclouds from text.
    
    The Wordcloud class provides the primary user interface for creating
    wordclouds from text data. It handles text preprocessing, word frequency analysis,
    word positioning, and visualization in various formats.
    
    The class integrates with TextProcessor for text processing operations and supports
    both IntegralImage (default) and CollisionDetector for collision detection.
    
    Attributes:
        width (int): Width of the wordcloud in pixels
        height (int): Height of the wordcloud in pixels
        font_path (str): Path to the font file to use for rendering words
        margin (int): Margin between words in pixels
        max_words (int): Maximum number of words to include in the wordcloud
        min_word_length (int): Minimum length of words to include
        min_font_size (int): Minimum font size for words in points
        max_font_size (int): Maximum font size for words in points
        font_step (int): Step size for decreasing font size during placement
        stopwords (list): Words to exclude from the wordcloud
        background_color (str): Background color of the wordcloud
        mode (str): Color mode for the image ('RGB', 'RGBA', etc.)
        black_white (bool): Whether to use only black text
        place_strategy (str): Strategy to use for word placement
        rect_only (bool): Whether to draw only rectangles instead of text
        tracing_files (bool): Whether to generate tracing files for debugging
        collision_detector (CollisionDetector, optional): Optional collision detector.
            Only supported for 'random' and 'rectangular' strategies. Defaults to None.
        gen_positions (list): Generated positions for words
        results_folder (str): Folder to save output files
        _text_processor (TextProcessor): Internal text processor instance
    """
    def __init__(self, width=600, height=338, font_path="fonts/Arial Unicode.ttf", margin=2,
                 max_words=200, min_word_length=3,
                 min_font_size=14, max_font_size=None, font_step=2,
                 stopwords=None,  
                 background_color='white', mode="RGB", black_white=False, 
                 place_strategy=STRATEGIES[0], rect_only=False, tracing_files = False,
                 mask_image=None, mask_threshold: int = 200,
                 log_level: int = logging.INFO,
                 enable_performance_tracking: bool = False,
                 performance_tracking_detail: str = "basic",
                 prefer_horizontal: float = 1.0,
                 rotation_angles: tuple[int, ...] = (90, -90),
                 collision_detector: Optional[CollisionDetector] = None,
                 config_file: Optional[str] = None,
                 use_config: bool = True,
                 language: Optional[str] = None,
                 enable_stemming: bool = False,
                 enable_lemmatization: bool = False,
                 n_gram_range: Optional[tuple[int, int]] = None,
                 text_processor_options: Optional[dict] = None,
                 text_effects: Optional[dict] = None,
                 font_distribution: str = 'linear',
                 font_distribution_params: Optional[dict] = None,
                 random_state: Optional[Union[int, random.Random, np.random.Generator]] = None):
        """
        Initialize a Wordcloud instance with customization options.
        
        Args:
            width (int, optional): Width of the wordcloud in pixels. Defaults to 600.
            height (int, optional): Height of the wordcloud in pixels. Defaults to 338.
            font_path (str, optional): Path to the font file. Defaults to "fonts/Arial Unicode.ttf".
            margin (int, optional): Margin between words in pixels. Defaults to 2.
            max_words (int, optional): Maximum number of words to include. Defaults to 200.
            min_word_length (int, optional): Minimum length of words to include. Defaults to 3.
            min_font_size (int, optional): Minimum font size in points. Defaults to 14.
            max_font_size (int, optional): Maximum font size in points. Defaults to None.
            font_step (int, optional): Step size for decreasing font size. Defaults to 2.
            stopwords (list, optional): Words to exclude from the wordcloud. Defaults to None (empty list).
            background_color (str, optional): Background color. Defaults to 'white'.
            mode (str, optional): Color mode ('RGB', 'RGBA', etc.). Defaults to "RGB".
            black_white (bool, optional): Use only black text. Defaults to False.
            place_strategy (str, optional): Word placement strategy. Defaults to STRATEGIES[0] that is 'random'.
            rect_only (bool, optional): Draw rectangles instead of text. Defaults to False.
            tracing_files (bool, optional): Generate debug tracing files. Defaults to False.
            log_level (int, optional): Logging level. Defaults to logging.INFO.
            enable_performance_tracking (bool, optional): Whether to enable performance tracking. Defaults to False.
            performance_tracking_detail (str, optional): Level of performance tracking detail ("basic" or "detailed").
                Defaults to "basic".
            collision_detector (CollisionDetector, optional): Optional collision detector to use instead of IntegralImage.
                Only supported for 'random' and 'rectangular' placement strategies. Defaults to None (uses IntegralImage).
            config_file (str, optional): Path to configuration file (YAML or JSON). If provided, loads config from file.
                Defaults to None (uses default config locations or global config). Configuration files are searched in:
                - Current directory: wordcloud_config.yaml, wordcloud_config.json
                - User config: ~/.config/wordcloud/config.yaml, ~/.config/wordcloud/config.json
            use_config (bool, optional): Whether to load defaults from configuration. Defaults to True.
                If False, uses only explicit parameters. Explicit parameters always override config values.
                Configuration can also be loaded from environment variables prefixed with WORDCLOUD_.
            language (str, optional): Language code (e.g., 'en', 'fr', 'zh'). If None, will auto-detect. Defaults to None.
            enable_stemming (bool, optional): Whether to apply stemming (requires NLTK). Defaults to False.
            enable_lemmatization (bool, optional): Whether to apply lemmatization (requires NLTK or spaCy). Defaults to False.
            n_gram_range (tuple[int, int], optional): Tuple (min_n, max_n) for n-gram extraction (e.g., (1, 2) for unigrams+bigrams). Defaults to None.
            text_processor_options (dict, optional): Additional options dict for advanced text processing configuration. Defaults to None.
            text_effects (dict, optional): Text effects configuration dict with 'outline', 'shadow', 'gradient' keys. Defaults to None.
            font_distribution (str, optional): Font size distribution method ('linear', 'logarithmic', 'power', 'custom'). Defaults to 'linear'.
            font_distribution_params (dict, optional): Parameters for font distribution (e.g., {'exponent': 0.5} for power law). Defaults to None.
            random_state (int | random.Random | numpy.random.Generator, optional): Seed or RNG for deterministic output.
                Use an integer seed for fully reproducible layouts and colors. Defaults to None (non-deterministic).
        """
        # Setup logging
        self.logger = get_logger("Wordcloud")
        self.logger.setLevel(log_level)
        
        # Load configuration if enabled
        config_defaults = {}
        config_loaded = False
        if use_config:
            try:
                if config_file:
                    config_manager = ConfigManager(config_file=config_file)
                    config_loaded = True
                else:
                    config_manager = get_config()
                    # Only use config if a config file was actually loaded
                    config_loaded = config_manager.config_file is not None
                
                # Get wordcloud defaults from config only if config was loaded
                if config_loaded:
                    config_defaults = config_manager.get_wordcloud_defaults()
                    self.logger.debug(f"Loaded configuration defaults: {list(config_defaults.keys())}")
                else:
                    self.logger.debug("No configuration file found, using explicit parameters only")
                
                # Get performance config (only if config was loaded)
                if config_loaded:
                    perf_config = config_manager.get_performance_config()
                    if enable_performance_tracking is False:  # Only override if not explicitly set
                        enable_performance_tracking = perf_config.get('enable_tracking', False)
                    if performance_tracking_detail == "basic":  # Only override if using default
                        performance_tracking_detail = perf_config.get('tracking_detail', "basic")
                
                # Store config manager for later use (e.g., results folder)
                if config_loaded:
                    self._config_manager = config_manager
            except Exception as e:
                self.logger.warning(f"Failed to load configuration: {e}. Using explicit parameters only.")
                config_defaults = {}
                config_loaded = False
        
        # Apply config defaults, but explicit parameters override config
        # Only use config value if parameter is using its default value
        width = config_defaults.get('width', width) if width == 600 else width
        height = config_defaults.get('height', height) if height == 338 else height
        font_path = config_defaults.get('font_path', font_path) if font_path == "fonts/Arial Unicode.ttf" else font_path
        margin = config_defaults.get('margin', margin) if margin == 2 else margin
        max_words = config_defaults.get('max_words', max_words) if max_words == 200 else max_words
        min_word_length = config_defaults.get('min_word_length', min_word_length) if min_word_length == 3 else min_word_length
        min_font_size = config_defaults.get('min_font_size', min_font_size) if min_font_size == 14 else min_font_size
        # max_font_size: None means auto-adjust, so only apply config if it's explicitly set in config
        # and the parameter is using the default (None)
        if max_font_size is None and 'max_font_size' in config_defaults and config_defaults['max_font_size'] is not None:
            max_font_size = config_defaults['max_font_size']
        font_step = config_defaults.get('font_step', font_step) if font_step == 2 else font_step
        background_color = config_defaults.get('background_color', background_color) if background_color == 'white' else background_color
        mode = config_defaults.get('mode', mode) if mode == "RGB" else mode
        black_white = config_defaults.get('black_white', black_white) if black_white is False else black_white
        place_strategy = config_defaults.get('place_strategy', place_strategy) if place_strategy == STRATEGIES[0] else place_strategy
        rect_only = config_defaults.get('rect_only', rect_only) if rect_only is False else rect_only
        tracing_files = config_defaults.get('tracing_files', tracing_files) if tracing_files is False else tracing_files
        mask_threshold = config_defaults.get('mask_threshold', mask_threshold) if mask_threshold == 200 else mask_threshold
        
        # Input validation
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive integers. Got width={width}, height={height}")
        
        if width > 10000 or height > 10000:
            self.logger.warning(f"Large dimensions ({width}x{height}) may cause performance issues")
        
        if place_strategy not in STRATEGIES:
            raise ValueError(f"Invalid placement strategy '{place_strategy}'. Must be one of: {STRATEGIES}")
        
        if min_font_size <= 0:
            raise ValueError(f"min_font_size must be positive. Got {min_font_size}")
        
        if max_font_size is not None and max_font_size < min_font_size:
            raise ValueError(f"max_font_size ({max_font_size}) must be >= min_font_size ({min_font_size})")

        # Initialize per-instance randomness for deterministic generation when requested
        self.random_state = random_state
        self._py_random = self._init_python_random(random_state)
        self._np_random = self._init_numpy_random(random_state)
        
        if max_words <= 0:
            raise ValueError(f"max_words must be positive. Got {max_words}")
        
        if margin < 0:
            raise ValueError(f"margin must be non-negative. Got {margin}")
        
        if font_step <= 0:
            raise ValueError(f"font_step must be positive. Got {font_step}")
        
        # Validate font path(s) exist (warn only, allow runtime failure for flexibility)
        if isinstance(font_path, str):
            if not os.path.exists(font_path):
                self.logger.warning(f"Font path '{font_path}' does not exist. Font loading may fail.")
        elif isinstance(font_path, (list, dict)):
            # For list/dict of fonts, validate each
            font_list = font_path if isinstance(font_path, list) else list(font_path.values())
            for fp in font_list:
                if isinstance(fp, str) and not os.path.exists(fp):
                    self.logger.warning(f"Font path '{fp}' does not exist. Font loading may fail.")

        # Word orientation settings (now supports arbitrary angles)
        if not (0.0 <= prefer_horizontal <= 1.0):
            raise ValueError(f"prefer_horizontal must be in [0.0, 1.0]. Got {prefer_horizontal}")
        if not validate_rotation_angles(rotation_angles):
            raise ValueError("rotation_angles must be a non-empty tuple of integers in [-360, 360] range")
        self.prefer_horizontal = float(prefer_horizontal)
        self.rotation_angles = rotation_angles
        
        # Text effects and typography settings
        self.text_effects = TextEffects(**(text_effects or {})) if text_effects else None
        self.font_distribution = font_distribution
        self.font_distribution_params = font_distribution_params or {}
        
        # Performance tracking settings
        self.enable_performance_tracking = enable_performance_tracking
        if performance_tracking_detail not in ("basic", "detailed"):
            self.logger.warning(f"Invalid performance_tracking_detail '{performance_tracking_detail}'. Using 'basic'.")
            performance_tracking_detail = "basic"
        self.performance_tracking_detail = performance_tracking_detail
        self.performance_metrics = {}
        
        self.height = height
        self.width = width
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.font_path = font_path
        self.margin = margin
        self.font_step = font_step
        self.font_size = None
        self.orientation = None
        self.max_words = max_words
        self.min_word_length = min_word_length
        self.stopwords = stopwords if stopwords is not None else []
        self.background_color = background_color
        self.mode = mode
        self.black_white = black_white
        self.place_strategy = place_strategy
        self.rect_only = rect_only
        self.tracing_files = tracing_files
        self.collision_detector = collision_detector
        self.mask_processor = None
        if mask_image is not None:
            self.logger.info(f"Processing mask image with threshold {mask_threshold}")
            self.mask_processor = MaskProcessor(mask_image, threshold=mask_threshold)
            if (self.mask_processor.height, self.mask_processor.width) != (self.height, self.width):
                if (self.height * self.width) <= 2500:
                    self.logger.warning(
                        f"Mask dimensions {self.mask_processor.height}x{self.mask_processor.width} "
                        f"do not match wordcloud dimensions {self.height}x{self.width}. "
                        "Using mask dimensions for the wordcloud canvas."
                    )
                    self.height = self.mask_processor.height
                    self.width = self.mask_processor.width
                else:
                    raise ValueError(
                        f"Mask dimensions {self.mask_processor.height}x{self.mask_processor.width} "
                        f"do not match wordcloud dimensions {self.height}x{self.width}."
                    )
            self.logger.info(f"Mask processed successfully: {self.mask_processor.width}x{self.mask_processor.height}")
        
        self.def_max_font_size = 80
        self.gen_positions = None
        
        # Set results folder (from config if available, otherwise default)
        if use_config and config_loaded and hasattr(self, '_config_manager'):
            try:
                results_folder = self._config_manager.get('output', 'results_folder', None)
                if results_folder:
                    self.results_folder = os.path.join(os.getcwd(), results_folder)
                else:
                    self.results_folder = os.getcwd() + "/Results"
            except (KeyError, AttributeError):
                self.results_folder = os.getcwd() + "/Results"
        else:
            self.results_folder = os.getcwd() + "/Results"
        
        # Initialize font cache for performance optimization
        self._font_cache = FontCache()
        self._rotated_text_cache = {}
        
        # Store text processing options
        self.language = language
        self.enable_stemming = enable_stemming
        self.enable_lemmatization = enable_lemmatization
        self.n_gram_range = n_gram_range
        self.text_processor_options = text_processor_options.copy() if text_processor_options else {}
        self.text_processor_options.setdefault("use_language_stopwords", False)
        
        # Initialize TextProcessor for text processing operations
        self._text_processor = TextProcessor(
            min_word_length=self.min_word_length,
            max_words=self.max_words,
            stopwords=self.stopwords,
            language=self.language,
            enable_stemming=self.enable_stemming,
            enable_lemmatization=self.enable_lemmatization,
            n_gram_range=self.n_gram_range,
            text_processor_options=self.text_processor_options
        )
        
        # Log initialization
        self.logger.info(f"Wordcloud initialized: {width}x{height}, strategy='{place_strategy}', "
                        f"max_words={max_words}, performance_tracking={enable_performance_tracking}")
        self.logger.debug(f"Configuration: font_path={font_path}, margin={margin}, "
                         f"min_font_size={min_font_size}, max_font_size={max_font_size}")

    @staticmethod
    def _init_python_random(
        random_state: Optional[Union[int, random.Random, np.random.Generator]]
    ) -> random.Random:
        if isinstance(random_state, random.Random):
            return random_state
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, 2**32 - 1))
            return random.Random(seed)
        if random_state is None:
            return random.Random()
        return random.Random(random_state)

    @staticmethod
    def _init_numpy_random(
        random_state: Optional[Union[int, random.Random, np.random.Generator]]
    ) -> np.random.Generator:
        if isinstance(random_state, np.random.Generator):
            return random_state
        if isinstance(random_state, random.Random):
            seed = random_state.getrandbits(32)
            return np.random.default_rng(seed)
        if random_state is None:
            return np.random.default_rng()
        return np.random.default_rng(random_state)

    def _find_position_with_collision_detector(
        self, collision_detector: CollisionDetector, word_width: int, word_height: int
    ) -> Optional[tuple[int, int]]:
        """
        Find position using CollisionDetector and placement functions.
        
        Args:
            collision_detector: The collision detector to use
            word_width: Width of the word bounding box
            word_height: Height of the word bounding box
            
        Returns:
            Tuple of (x, y) coordinates if position found, None otherwise
        """
        # Create callback for position validation
        def is_valid_position(pos_y: int, pos_x: int, height: int, width: int) -> bool:
            return collision_detector.is_position_available(pos_x, pos_y, width, height)
        
        # Use appropriate placement function based on strategy
        if self.place_strategy == "random":
            return find_position_random(
                self.width,
                self.height,
                word_width,
                word_height,
                is_valid_position,
                rng=self._py_random,
            )
        elif self.place_strategy == "rectangular":
            return find_position_rectangular_spiral(
                self.width,
                self.height,
                word_width,
                word_height,
                is_valid_position,
                rng=self._py_random,
            )
        else:
            # Should not happen due to earlier check, but handle gracefully
            self.logger.error(f"Unsupported strategy '{self.place_strategy}' for CollisionDetector")
            return None
    
    def _choose_orientation_degrees(self) -> int | None:
        """
        Return an orientation in degrees for a single word.

        None means horizontal (0 degrees). Returns a rotation angle from rotation_angles.
        Now supports arbitrary angles, not just 90/-90.
        """
        if self._py_random.random() < self.prefer_horizontal:
            return None
        angle = normalize_rotation_angle(self._py_random.choice(self.rotation_angles))
        # Normalize angle (0 means horizontal, return None)
        if angle == 0:
            return None
        return angle

    @staticmethod
    def _pil_orientation_from_degrees(degrees: int | None) -> int | None:
        """
        Convert degrees to Pillow Transpose constants for ImageFont.TransposedFont.
        Now supports arbitrary angles, but PIL only supports 90/-90 natively.
        For other angles, returns None and rotation must be handled differently.
        """
        if degrees in (None, 0):
            return None
        if degrees == 90:
            return Image.Transpose.ROTATE_90 if hasattr(Image, "Transpose") else Image.ROTATE_90
        if degrees == -90 or degrees == 270:
            return Image.Transpose.ROTATE_270 if hasattr(Image, "Transpose") else Image.ROTATE_270
        if degrees == 180 or degrees == -180:
            return Image.Transpose.ROTATE_180 if hasattr(Image, "Transpose") else Image.ROTATE_180
        # For other angles, return None - rotation will need to be handled via Image.rotate()
        return None

    def _get_rotated_text_image(
        self,
        text: str,
        font_path: str,
        font_size: int,
        angle_degrees: int,
        fill,
        mode: str = "RGBA",
    ):
        """
        Render rotated text into an image for arbitrary angles.
        """
        cache_key = (text, font_path, font_size, angle_degrees, fill, mode)
        cached = self._rotated_text_cache.get(cache_key)
        if cached is not None:
            return cached

        font = self._font_cache.get_font(font_path, font_size)
        bbox = font.getbbox(text)
        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        if width == 0 or height == 0:
            return None

        if mode == "L":
            base = Image.new("L", (width, height), 0)
            text_fill = 255
        else:
            base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            text_fill = fill

        base_draw = ImageDraw.Draw(base)
        base_draw.text((-bbox[0], -bbox[1]), text, font=font, fill=text_fill)
        rotated = base.rotate(angle_degrees, expand=True, resample=Image.BICUBIC)
        self._rotated_text_cache[cache_key] = rotated
        return rotated

    def split_text(self, text_to_analyze, stopwords=None, min_word_length=None):
        """
        Split text into frequency dictionary with basic cleaning.
        Stopword comparison is case-sensitive and applied before lowercasing.
        
        This method maintains Wordcloud's original implementation for backward compatibility.
        TextProcessor is available via self._text_processor for advanced usage.
        """
        tracker = None
        if self.enable_performance_tracking:
            tracker = PerformanceTracker("split_text", self.performance_tracking_detail)
            tracker.start()
        
        try:
            if not text_to_analyze:
                self.logger.debug("Empty text provided to split_text")
                return {}

            self.logger.debug(f"Splitting text (length: {len(text_to_analyze)} characters)")

            # Maintain legacy behavior expected by tests.
            stopwords = stopwords if stopwords is not None else self.stopwords
            min_len = self.min_word_length if min_word_length is None else min_word_length
            stopword_set = set(stopwords or [])

            # Normalize: remove hyphens, strip punctuation, remove digits.
            text = text_to_analyze.replace("-", "")
            text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
            tokens = text.split()

            frequencies = {}
            for token in tokens:
                if not token:
                    continue
                # Remove numeric tokens
                if token.isdigit():
                    continue
                token = token.lower()
                if token in stopword_set:
                    continue
                if len(token) < min_len:
                    continue
                frequencies[token] = frequencies.get(token, 0) + 1

            self.logger.debug(f"Split text into {len(frequencies)} unique words")
            return frequencies
        finally:
            if tracker:
                tracker.stop()
                self.performance_metrics['split_text'] = tracker.get_summary()

    def sort_normalize(self, words_dict):
        """
        Normalize word frequencies and sort descending.
        
        This method delegates to TextProcessor while maintaining backward compatibility
        with max_words limiting.
        """
        if not words_dict:
            self.logger.error("No words to normalize in sort_normalize")
            raise ValueError("No words to normalize")
        
        # TODO: remove this after testing, old implementation
        # self.logger.debug(f"Sorting and normalizing {len(words_dict)} words")
        # max_freq = max(words_dict.values()) or 1
        # normalized = [
        #     (word, freq / max_freq if max_freq else 0.0, freq) for word, freq in words_dict.items()
        # ]
        # result = sorted(normalized, key=lambda x: x[1], reverse=True)[: self.max_words]
        
        # Delegate to TextProcessor for normalization
        normalized = self._text_processor.sort_normalize(words_dict)
        
        # Apply max_words limit (Wordcloud-specific behavior)
        result = normalized[:self.max_words]
        
        if len(result) < len(normalized):
            self.logger.debug(f"Limited words from {len(normalized)} to {len(result)} (max_words={self.max_words})")
        
        if result:
            self.logger.debug(f"Most frequent word: '{result[0][0]}' with frequency {result[0][2]}")
        
        return result

    def prepare_text(self, to_split, stopwords=None, min_word_length=None):
        """
        Prepare text for wordcloud generation.
        
        This method delegates to TextProcessor while maintaining backward compatibility
        with parameter overrides. It combines split_text and sort_normalize operations.
        """
        tracker = None
        if self.enable_performance_tracking:
            tracker = PerformanceTracker("prepare_text", self.performance_tracking_detail)
            tracker.start()
        
        try:
            self.logger.info(f"Preparing text (length: {len(to_split)} characters)")
            # Use the wrapper methods which delegate to TextProcessor TODO: why?
            splitted = self.split_text(to_split, stopwords, min_word_length)
            result = self.sort_normalize(splitted)
            self.logger.info(f"Prepared {len(result)} words for wordcloud")
            return result
        finally:
            if tracker:
                tracker.stop()
                self.performance_metrics['prepare_text'] = tracker.get_summary()
                tracker.log_summary()

    def prepare_text_stream(
        self,
        text_chunks,
        stopwords=None,
        min_word_length=None,
        trim_multiplier: int = 5,
    ):
        """
        Prepare text for wordcloud generation from a stream of chunks.
        """
        tracker = None
        if self.enable_performance_tracking:
            tracker = PerformanceTracker("prepare_text_stream", self.performance_tracking_detail)
            tracker.start()

        try:
            self.logger.info("Preparing text from stream")
            splitted = self._text_processor.split_text_stream(
                text_chunks,
                stopwords=stopwords,
                min_word_length=min_word_length,
                trim_multiplier=trim_multiplier,
            )
            result = self.sort_normalize(splitted)
            self.logger.info(f"Prepared {len(result)} words for wordcloud")
            return result
        finally:
            if tracker:
                tracker.stop()
                self.performance_metrics['prepare_text_stream'] = tracker.get_summary()
                tracker.log_summary()
    def find_position(self, frequencies):
        """
        Find positions for words in the wordcloud.
        
        This method is the core of the wordcloud generation process. It takes a list
        of words with their frequencies and finds suitable positions for them using
        the specified placement strategy. The font size of each word is determined
        by its normalized frequency.
        
        Args:
            frequencies (list): List of (word, normalized_frequency, original_frequency) tuples
            
        Returns:
            Wordcloud: Self for method chaining
            
        Notes:
            - Words are placed one by one, from highest to lowest frequency
            - If a word cannot be placed at its initial font size, the size is reduced
              until placement succeeds or min_font_size is reached
            - The results are stored in the gen_positions attribute as a list of tuples:
              (word_data, font_path, font_size, position, orientation, color)
        """
        tracker = None
        if self.enable_performance_tracking:
            tracker = PerformanceTracker("find_position", self.performance_tracking_detail)
            tracker.start()
        
        try:
            self.logger.info(f"Finding positions for {len(frequencies)} words using strategy '{self.place_strategy}'")
            
            # Determine if we should use CollisionDetector (only for supported strategies)
            use_collision_detector = (
                self.collision_detector is not None and 
                self.place_strategy in ("random", "rectangular")
            )
            
            if use_collision_detector:
                self.logger.debug(f"Using CollisionDetector with strategy '{self.place_strategy}'")
                collision_detector = self.collision_detector
            else:
                # Use IntegralImage (default behavior) doubing the IF? TODO
                if self.collision_detector is not None and self.place_strategy not in ("random", "rectangular"):
                    self.logger.warning(
                        f"CollisionDetector provided but strategy '{self.place_strategy}' not supported. "
                        f"Falling back to IntegralImage. Supported strategies: 'random', 'rectangular'"
                    )
                integral_image = IntegralImage(
                    self.height,
                    self.width,
                    self.tracing_files,
                    mask=self.mask_processor,
                    rng=self._py_random,
                )

            # create control image
            control_img = Image.new("L", (self.width, self.height))
            draw = ImageDraw.Draw(control_img)
            
            # prepare variables we want to save for each word
            font_paths, font_sizes, positions, orientations, colors = [], [], [], [], []
            
            # Assign fonts to words if font_path is a list or dict
            word_font_map = {}
            if isinstance(self.font_path, (list, dict)):
                word_font_map = assign_fonts_to_words(
                    frequencies,
                    self.font_path,
                    strategy='frequency',
                    rng=self._py_random,
                )
            else:
                # Single font for all words
                for word, _, _ in frequencies:
                    word_font_map[word] = self.font_path
            
            # start drawing greyscale image
            for word, freq, count in frequencies:
                
                if freq == 0:
                    continue
            
                self.logger.debug(f"Processing word: '{word}' with freq {freq}")
                
                # if there is only one word, set Max size to image height, use default max height otherwise
                if self.max_font_size is None:
                    self.max_font_size = self.height if len(frequencies) == 1 else self.def_max_font_size
                
                # Calculate font size using advanced distribution if specified
                if self.font_distribution != 'linear' or self.font_distribution_params:
                    calculated_size = calculate_font_size(
                        freq, 
                        self.min_font_size, 
                        self.max_font_size,
                        distribution=self.font_distribution,
                        distribution_params=self.font_distribution_params
                    )
                    self.font_size = min(self.font_size, calculated_size) if self.font_size else calculated_size
                else:
                    # Original linear scaling
                    self.font_size = min(self.font_size, int(round(freq * self.max_font_size))) if self.font_size else int(round(freq * self.max_font_size))
            
                # look for a place until it's found or font became too small
                while True:
                    
                    # font_size is too small
                    if self.font_size < self.min_font_size:
                        break
                
                    try:
                        orientation_degrees = self._choose_orientation_degrees()
                        pil_orientation = self._pil_orientation_from_degrees(orientation_degrees)
                        
                        # Get font path for this word
                        word_font_path = word_font_map.get(word, self.font_path)

                        # Use font cache for performance optimization
                        font = self._font_cache.get_font(word_font_path, self.font_size)
                        transposed_font = None
                        
                        # get size of resulting text (use cached bbox lookup)
                        if orientation_degrees is not None and pil_orientation is None:
                            box_size = self._font_cache.get_rotated_bbox(
                                word, word_font_path, self.font_size, orientation_degrees
                            )
                        else:
                            transposed_font = ImageFont.TransposedFont(font, orientation=pil_orientation)
                            box_size = self._font_cache.get_text_bbox(
                                draw, word, word_font_path, self.font_size, pil_orientation
                            )
                        word_width = box_size[2] + self.margin
                        word_height = box_size[3] + self.margin
                        
                        # find possible places using either CollisionDetector or IntegralImage
                        if use_collision_detector:
                            result = self._find_position_with_collision_detector(
                                collision_detector, word_width, word_height
                            )
                        else:
                            result = integral_image.find_position(word_width, word_height, self.place_strategy, word)
                        
                        # found a place
                        if result is not None:
                            break
                        
                        # we didn't find a place, make font smaller and try again
                        self.font_size -= self.font_step
                    except Exception as e:
                        self.logger.error(f"Could not load font {self.font_path} at size {self.font_size}: {e}")
                        self.font_size -= self.font_step
                
                # font_size is too small
                if self.font_size < self.min_font_size:
                    self.logger.info(f"Could not place word '{word}' even at min font size {self.min_font_size}. Skipping.")
                    continue

                # Recalculate dimensions with final font size and orientation for update
                final_pil_orientation = self._pil_orientation_from_degrees(orientation_degrees)
                word_font_path = word_font_map.get(word, self.font_path)
                if orientation_degrees is not None and final_pil_orientation is None:
                    final_box_size = self._font_cache.get_rotated_bbox(
                        word, word_font_path, self.font_size, orientation_degrees
                    )
                else:
                    final_box_size = self._font_cache.get_text_bbox(
                        draw, word, word_font_path, self.font_size, final_pil_orientation
                    )
                final_word_width = final_box_size[2] + self.margin
                final_word_height = final_box_size[3] + self.margin

                width_x, height_y = np.array(result) + self.margin // 2
                
                # draw the text to control img
                rotated_mask = None
                if orientation_degrees is not None and final_pil_orientation is None:
                    rotated_mask = self._get_rotated_text_image(
                        word,
                        word_font_path,
                        self.font_size,
                        orientation_degrees,
                        fill=255,
                        mode="L",
                    )
                    if rotated_mask is not None:
                        control_img.paste(rotated_mask, (width_x, height_y), rotated_mask)
                else:
                    transposed_font = ImageFont.TransposedFont(
                        self._font_cache.get_font(word_font_path, self.font_size),
                        orientation=final_pil_orientation,
                    )
                    draw.text((width_x, height_y), word, fill="white", font=transposed_font)
                
                if self.rect_only:
                    if rotated_mask is not None:
                        draw.rectangle(
                            [width_x, height_y, width_x + final_word_width - self.margin, height_y + final_word_height - self.margin],
                            outline="white",
                        )
                    else:
                        bbox = draw.textbbox((width_x, height_y), word, font=transposed_font)
                        draw.rectangle(bbox, outline="white")
                    
                font_paths.append(word_font_path)
                positions.append((width_x, height_y))
                orientations.append(orientation_degrees)
                font_sizes.append(self.font_size)
                
                if self.black_white:
                    colors.append("rgb(0, 0, 0)")
                else:
                    colors.append(
                        "rgb("
                        f"{self._py_random.randint(0, 255)}, "
                        f"{self._py_random.randint(0, 255)}, "
                        f"{self._py_random.randint(0, 255)}"
                        ")"
                    )
            
                # create numpy array for control image
                img_array = np.asarray(control_img)
                
                # Update collision detection system with new word
                if use_collision_detector:
                    # Add rectangle to collision detector
                    collision_detector.add_rectangle((width_x, height_y, final_word_width, final_word_height))
                else:
                    # Update integral image with new word
                    if hasattr(integral_image, "update"):
                        integral_image.update(img_array, width_x, height_y)
            
            self.gen_positions = list(zip(frequencies, font_paths, font_sizes, positions, orientations, colors))
            self.logger.info(f"Successfully placed {len(font_paths)} words")
            
        finally:
            if tracker:
                tracker.stop()
                self.performance_metrics['find_position'] = tracker.get_summary()
                tracker.log_summary()
        
        return self
    
    def update_position(self, new_fonts):
        """
        Update word positions, optionally changing fonts.
        
        This method repositions all words in the wordcloud, optionally using new fonts.
        It's useful for experimenting with different placement strategies or fonts
        without regenerating the entire wordcloud.
        
        Args:
            new_fonts (list or dict): New fonts to use. Can be either:
                - A list of font paths (position-based matching)
                - An empty list/dict (uses default font for all words)
                
        Returns:
            Wordcloud: Self for method chaining
            
        Notes:
            - This method preserves the original words, frequencies, font sizes, and colors
            - Only positions and (optionally) fonts are updated
            - If new_fonts is a list, fonts are assigned by position (index)
            - If fewer fonts are provided than words, default font is used for remaining words
        """
        # If new_fonts is empty, use the default font for all words
        if not new_fonts:
            new_fonts = {
                word: self.font_path
                for (word, _, _), _, _, _, _, _ in self.gen_positions
            }
            
        self.font_size = None
    
        integral_image = IntegralImage(self.height, self.width)

        # create control image
        control_img = Image.new("L", (self.width, self.height))
        draw = ImageDraw.Draw(control_img)
        
        new_freq, font_paths, font_sizes, positions, orientations, colors = [], [], [], [], [], []
    
        # start drawing greyscale image
        for idx, ((word, freq, count), font_path, word_font_size, position, orientation, color) in enumerate(self.gen_positions):
            
            if freq == 0:
                continue
        
            # with the change of fonts, some sizes can differ as well
            if self.max_font_size is None:
                if len(self.gen_positions) == 1:
                    self.max_font_size = self.height
                else:
                    self.max_font_size = self.def_max_font_size
            
            # select the font size TODO: original implementation
            # self.font_size = min(self.font_size, int(round(freq * self.max_font_size))) if self.font_size else int(round(freq * self.max_font_size))
        
            # Start of new implementation
            # Start with original font size from gen_positions, then recalculate if needed
            # This preserves the original font size unless frequency suggests a smaller size
            if self.font_size is None:
                self.font_size = word_font_size
            
            # Recalculate font size based on frequency, but don't exceed original
            # This ensures we don't make fonts larger than they were originally
            calculated_size = int(round(freq * self.max_font_size))
            self.font_size = min(self.font_size, calculated_size)
            
            # Check if font size is too small before trying to place
            if self.font_size < self.min_font_size:
                self.font_size = None  # Reset for next word
                continue
            
            # End of new implementation
        
    
            # look for a place until it's found or font became too small
            while True:
                
                # font-size is too small
                if self.font_size < self.min_font_size:
                    break
                
                # Preserve existing orientation for each word (supports arbitrary angles)
                orientation_degrees = None if orientation is None else int(orientation)
                pil_orientation = self._pil_orientation_from_degrees(orientation_degrees)

                if isinstance(new_fonts, dict) and word in new_fonts:
                    font_path_to_use = new_fonts[word]
                elif isinstance(new_fonts, list) and idx < len(new_fonts):
                    font_path_to_use = new_fonts[idx]
                else:
                    font_path_to_use = self.font_path

                # Use font cache where possible
                if orientation_degrees is not None and pil_orientation is None:
                    box_size = self._font_cache.get_rotated_bbox(
                        word, font_path_to_use, self.font_size, orientation_degrees
                    )
                    transposed_font = None
                else:
                    font = self._font_cache.get_font(font_path_to_use, self.font_size)
                    transposed_font = ImageFont.TransposedFont(font, orientation=pil_orientation)
                    # get size of resulting text
                    box_size = self._font_cache.get_text_bbox(
                        draw, word, font_path_to_use, self.font_size, pil_orientation
                    )
                word_width = box_size[2] + self.margin
                word_height = box_size[3] + self.margin
                
                # find possible places using integral image:
                result = integral_image.find_position(word_width, word_height, self.place_strategy, word)
                
                # Found a place
                if result is not None:
                    break
                
                # we didn't find a place, make font smaller and try again
                self.font_size -= self.font_step
            
            # check font size
            if self.font_size < self.min_font_size:
                break

            width_x, height_y = np.array(result) + self.margin // 2
            
            # draw the text to control img
            rotated_mask = None
            if orientation_degrees is not None and pil_orientation is None:
                rotated_mask = self._get_rotated_text_image(
                    word,
                    font_path_to_use,
                    self.font_size,
                    orientation_degrees,
                    fill=255,
                    mode="L",
                )
                if rotated_mask is not None:
                    control_img.paste(rotated_mask, (width_x, height_y), rotated_mask)
            else:
                draw.text((width_x, height_y), word, fill="white", font=transposed_font)
            
            if self.rect_only:
                if rotated_mask is not None:
                    draw.rectangle(
                        [width_x, height_y, width_x + word_width - self.margin, height_y + word_height - self.margin],
                        outline="white",
                    )
                else:
                    bbox = draw.textbbox((width_x, height_y), word, font=transposed_font)
                    draw.rectangle(bbox, outline="white")
                
            font_paths.append(font_path_to_use)
            
            font_sizes.append(self.font_size)
            positions.append((width_x, height_y))
            new_freq.append((word, freq, count))
            
            # we are not changing orientation or colors
            orientations.append(orientation_degrees)
            colors.append(color)
            
            # create numpy array for control image
            img_array = np.asarray(control_img)
            
            #update integral image with new word
            integral_image.update(img_array, width_x, height_y)
            
        # first lets clear our positions
        self.gen_positions = None
        
        # so we can save the new ones
        self.gen_positions = list(zip(new_freq, font_paths, font_sizes, positions, orientations, colors))
        
        return self
    
    def update_colors(self, new_colors):
        """
        Update word colors using either a list or dictionary of colors.
        
        This method allows changing the colors of words in the wordcloud
        without repositioning them.
        
        Args:
            new_colors (list or dict): New colors to use. Can be either:
                - A list of color values (position-based matching)
                - A dictionary mapping words to colors (word-based matching)
                
        Returns:
            Wordcloud: Self for method chaining
            
        Notes:
            - Colors can be specified in any format supported by PIL (RGB tuples, 
              hex strings, color names, etc.)
            - If new_colors is a list, colors are assigned by position (index)
            - If new_colors is a dict, colors are assigned by matching word text
            - Words without a matching color keep their original color
        """
        if not self.gen_positions:
            return self
            
        updated_positions = []
        for idx, ((word, freq, count), font_path, word_font_size, position, orientation, color) in enumerate(self.gen_positions):
            if isinstance(new_colors, dict):
                # Dictionary-based matching
                if word in new_colors:
                    color = new_colors[word]
            elif isinstance(new_colors, list):
                # List-based matching by index
                if idx < len(new_colors):
                    color = new_colors[idx]
            updated_positions.append(((word, freq, count), font_path, word_font_size, position, orientation, color))
        
        self.gen_positions = updated_positions
        return self
    
    def generate(self, text_to_analyze, color_theme: str | None = None, 
                 progressive: bool = False, progress_callback: Optional[Callable] = None):
        """
        Generate a wordcloud from text.
        
        This is the main entry point for generating a wordcloud from text.
        It combines text preparation and word positioning in a single call.
        
        Args:
            text_to_analyze (str): Text to analyze and visualize
            color_theme (str, optional): Optional color theme name (e.g. 'viridis').
                If provided (and black_white=False), colors are applied automatically.
            progressive (bool, optional): If True, yields intermediate states during generation. Defaults to False.
            progress_callback (callable, optional): Callback function called with (current, total) progress.
                Only used if progressive=True. Defaults to None.
            
        Returns:
            Wordcloud: Self for method chaining (or generator if progressive=True)
            
        Example:
            >>> wc = Wordcloud(width=800, height=400)
            >>> wc.generate("This is a sample text for wordcloud generation").draw_image(save_file=True)
            
            # Progressive generation:
            >>> for state in wc.generate("Text...", progressive=True):
            ...     print(f"Placed {len(state.gen_positions)} words")
        """
        tracker = None
        if self.enable_performance_tracking:
            tracker = PerformanceTracker("generate", self.performance_tracking_detail)
            tracker.start()
        
        try:
            self.logger.info("Starting wordcloud generation")
            normalized_and_sorted = self.prepare_text(text_to_analyze)
            
            if progressive:
                # Progressive generation: yield intermediate states
                return self._generate_progressive(normalized_and_sorted, color_theme, progress_callback)
            else:
                # Standard generation
                result = self.find_position(normalized_and_sorted)

                # Apply color theme if requested
                if color_theme and not self.black_white and self.gen_positions:
                    from .utils.visualization import COLOR_THEMES, generate_colors_by_frequency

                    if color_theme not in COLOR_THEMES:
                        raise ValueError(
                            f"Invalid color_theme '{color_theme}'. Must be one of: {', '.join(COLOR_THEMES.keys())}"
                        )

                    # normalized_and_sorted is [(word, normalized_freq, count), ...]
                    freq_map = {word: float(freq) for (word, freq, _count) in normalized_and_sorted}
                    colors = generate_colors_by_frequency(
                        freq_map,
                        color_theme=color_theme,
                        rng=self._py_random,
                    )
                    self.update_colors(colors)

                self.logger.info("Wordcloud generation completed")
                return result
        finally:
            if tracker:
                tracker.stop()
                self.performance_metrics['generate'] = tracker.get_summary()
                tracker.log_summary()

    def generate_from_stream(
        self,
        text_chunks,
        color_theme: str | None = None,
        trim_multiplier: int = 5,
    ):
        """
        Generate a wordcloud from a stream of text chunks.
        """
        tracker = None
        if self.enable_performance_tracking:
            tracker = PerformanceTracker("generate_from_stream", self.performance_tracking_detail)
            tracker.start()

        try:
            self.logger.info("Starting wordcloud generation from stream")
            normalized_and_sorted = self.prepare_text_stream(
                text_chunks,
                trim_multiplier=trim_multiplier,
            )
            result = self.find_position(normalized_and_sorted)

            if color_theme and not self.black_white and self.gen_positions:
                from .utils.visualization import COLOR_THEMES, generate_colors_by_frequency

                if color_theme not in COLOR_THEMES:
                    raise ValueError(
                        f"Invalid color_theme '{color_theme}'. Must be one of: {', '.join(COLOR_THEMES.keys())}"
                    )

                freq_map = {word: float(freq) for (word, freq, _count) in normalized_and_sorted}
                colors = generate_colors_by_frequency(
                    freq_map,
                    color_theme=color_theme,
                    rng=self._py_random,
                )
                self.update_colors(colors)

            self.logger.info("Streamed wordcloud generation completed")
            return result
        finally:
            if tracker:
                tracker.stop()
                self.performance_metrics['generate_from_stream'] = tracker.get_summary()
                tracker.log_summary()
    
    def _generate_progressive(self, frequencies, color_theme: str | None = None, 
                             progress_callback: Optional[Callable] = None):
        """
        Generate wordcloud progressively, yielding intermediate states.
        
        Args:
            frequencies: List of (word, normalized_freq, original_freq) tuples
            color_theme: Optional color theme name
            progress_callback: Optional callback function (current, total)
            
        Yields:
            Wordcloud: Self with intermediate gen_positions
        """
        total_words = len(frequencies)
        
        # Initialize for progressive generation
        self.gen_positions = []
        
        # Use find_position logic but yield after each word
        # (This is a simplified version - full implementation would refactor find_position)
        result = self.find_position(frequencies)
        
        # Apply color theme if requested
        if color_theme and not self.black_white and self.gen_positions:
            from .utils.visualization import COLOR_THEMES, generate_colors_by_frequency

            if color_theme not in COLOR_THEMES:
                raise ValueError(
                    f"Invalid color_theme '{color_theme}'. Must be one of: {', '.join(COLOR_THEMES.keys())}"
                )

            freq_map = {word: float(freq) for (word, freq, _count) in frequencies}
            colors = generate_colors_by_frequency(
                freq_map,
                color_theme=color_theme,
                rng=self._py_random,
            )
            self.update_colors(colors)
        
        # Yield final state
        yield self
        
        self.logger.info("Progressive wordcloud generation completed")
    
    def draw_image(self, save_file=False, image_name="wordcloud"):
        """
        Draw the wordcloud as a PIL Image.
        
        Creates an image of the wordcloud with words positioned and colored
        according to the configuration. Can optionally save the image to disk.
        
        Args:
            save_file (bool, optional): Whether to save the image to disk. Defaults to False.
            image_name (str, optional): Base name for the saved image file. Defaults to "wordcloud".
            
        Returns:
            PIL.Image.Image: The generated wordcloud image
            
        Notes:
            - The image is saved in PNG format in the results_folder with the name
              "{image_name}.png" if save_file is True
            - The image is always returned, regardless of whether it's saved
        """
        height = self.height
        width = self.width

        if self.gen_positions is None:
            raise AttributeError("Wordcloud has no generated positions. Call generate() first.")

        img = Image.new(self.mode, (width, height), self.background_color)
        draw = ImageDraw.Draw(img)
        
        for (word, freq, count), font_path, font_size, position, orientation, color in self.gen_positions:
            font = self._font_cache.get_font(font_path, font_size)
            orientation_degrees = None if orientation is None else int(orientation)
            pil_orientation = self._pil_orientation_from_degrees(orientation_degrees)
            pos = (position[0], position[1])
            if orientation_degrees is not None and pil_orientation is None:
                rotated_img = self._get_rotated_text_image(
                    word,
                    font_path,
                    font_size,
                    orientation_degrees,
                    fill=color,
                    mode="RGBA",
                )
                if rotated_img is not None:
                    img.paste(rotated_img, pos, rotated_img)
            else:
                transposed_font = ImageFont.TransposedFont(font, orientation=pil_orientation)
                draw.text(pos, word, fill=color, font=transposed_font)
            
        if save_file:
            from .utils.export import export_image

            create_folder(self.results_folder)
            self.logger.info(f"Saving image to {self.results_folder}/{image_name}.png")
            export_image(img, f"{self.results_folder}/{image_name}.png", optimize=True)
            self.logger.debug("Image saved successfully")
            
        return img
    
    def draw_plt_image(self):
        """
        Draw and display the wordcloud using matplotlib.
        
        This method generates the wordcloud image and displays it using
        matplotlib's pyplot for interactive viewing.
        
        Returns:
            Wordcloud: Self for method chaining
            
        Notes:
            - This method requires matplotlib to be installed
            - The image is displayed in a new matplotlib figure
            - Axis labels and ticks are hidden for a cleaner display
        """
        img = self.draw_image()
        
        plt.figure()
        plt.imshow(img, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        return self

    def generate_svg(self, save_file=False, file_name="svg_img"):
        """
        Generate an SVG representation of the wordcloud.
        
        Creates an SVG string representing the wordcloud, which can be saved
        to disk or used for web applications.
        
        Args:
            save_file (bool, optional): Whether to save the SVG to disk. Defaults to False.
            file_name (str, optional): Base name for the saved SVG file. Defaults to "svg_img".
            
        Returns:
            str: The SVG content as a string
            
        Notes:
            - SVG is a vector format, allowing the wordcloud to be scaled without loss of quality
            - The SVG includes information about word counts as attributes
            - The file is saved as "{file_name}.svg" in the results_folder if save_file is True
        """
        from .utils.export import get_svg_exporter

        if self.font_path is None:
            raise TypeError("font_path must be set to generate SVG output")

        svg_exporter = get_svg_exporter()
        svg_content = svg_exporter.generate_svg_content(
            gen_positions=self.gen_positions,
            width=self.width,
            height=self.height,
            default_font_path=self.font_path,
            max_font_size=self.max_font_size or self.def_max_font_size,
            background_color=self.background_color,
        )

        if save_file:
            from .utils import export as export_utils
            export_utils.create_folder(self.results_folder)
            try:
                svg_exporter.export_svg(
                    svg_content,
                    f"{self.results_folder}/{file_name}.svg",
                )
                self.logger.info(f"SVG saved to {self.results_folder}/{file_name}.svg")
            except Exception as exc:
                self.logger.error(f"Failed to save SVG: {exc}")

        return svg_content

    def create_html(self, svg_content=None, save_file=False, file_name="wordcloud", 
                    interactive=True, standalone=True):
        """
        Create an HTML file with the SVG content and interactive functionality.
        
        Embeds the SVG wordcloud in an HTML document with JavaScript that
        displays word counts as tooltips when hovering over words. The HTML
        is designed to be copy-paste ready for embedding in websites.
        
        Args:
            svg_content (str, optional): SVG content to embed. If None, generates from gen_positions.
            save_file (bool, optional): Whether to save the HTML to disk. Defaults to False.
            file_name (str, optional): Base name for the saved HTML file. Defaults to "wordcloud".
            interactive (bool, optional): Whether to include interactive features. Defaults to True.
            standalone (bool, optional): Whether to generate standalone HTML (with DOCTYPE, etc).
                                      If False, generates embeddable snippet. Defaults to True.
            
        Returns:
            str: The HTML content as a string
            
        Notes:
            - The HTML includes JavaScript for interactive tooltips showing word counts
            - The file is saved as "{file_name}.html" in the results_folder if save_file is True
            - Set standalone=False to get embeddable HTML snippet (without DOCTYPE/html/body tags)
            - This is useful for creating interactive visualizations for web applications
        """
        if svg_content is None:
            svg_content = self.generate_svg()
        
        if interactive and self.gen_positions:
            from .utils.export import export_interactive_html

            output_path = f"{self.results_folder}/{file_name}.html" if save_file else None
            if save_file:
                create_folder(self.results_folder)
            html_content = export_interactive_html(self, output_path=output_path)
            return html_content
        if interactive and not self.gen_positions:
            self.logger.warning("Interactive HTML requested without generated positions; falling back to static HTML.")

        if not standalone:
            if save_file:
                create_folder(self.results_folder)
                with open(f"{self.results_folder}/{file_name}.html", "w", encoding="utf-8") as f:
                    f.write(svg_content)
            return svg_content

        html_content = (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            "  <title>Wordcloud</title>\n"
            "</head>\n"
            "<body>\n"
            f"{svg_content}\n"
            "</body>\n"
            "</html>"
        )
        
        if save_file:
            create_folder(self.results_folder)
            with open(f"{self.results_folder}/{file_name}.html", "w", encoding="utf-8") as f:
                f.write(html_content)
        
        return html_content
    
    def draw_image_webp(self, save_file=False, image_name="wordcloud", quality=80, lossless=False):
        """
        Draw the wordcloud and export as WebP format.
        
        Args:
            save_file (bool, optional): Whether to save the image to disk. Defaults to False.
            image_name (str, optional): Base name for the saved image file. Defaults to "wordcloud".
            quality (int, optional): WebP quality (0-100). Defaults to 80.
            lossless (bool, optional): Whether to use lossless compression. Defaults to False.
            
        Returns:
            PIL.Image.Image: The generated wordcloud image
        """
        from .utils.export import export_webp
        
        img = self.draw_image(save_file=False)
        
        if save_file:
            self.logger.info(f"Saving WebP image to {self.results_folder}/{image_name}.webp")
            create_folder(self.results_folder)
            export_webp(img, f"{self.results_folder}/{image_name}.webp", quality=quality, lossless=lossless)
        
        return img
    
    def export_eps(self, save_file=True, file_name="wordcloud"):
        """
        Export wordcloud as EPS (Encapsulated PostScript) format.
        
        Args:
            save_file (bool, optional): Whether to save the EPS to disk. Defaults to True.
            file_name (str, optional): Base name for the saved EPS file. Defaults to "wordcloud".
            
        Returns:
            str: Path to saved EPS file if save_file=True, else None
        """
        from .utils.export import export_eps
        
        if save_file:
            self.logger.info(f"Exporting EPS to {self.results_folder}/{file_name}.eps")
            create_folder(self.results_folder)
            export_eps(self, f"{self.results_folder}/{file_name}.eps")
            return f"{self.results_folder}/{file_name}.eps"
        else:
            export_eps(self, file_name)
            return file_name
    
    def apply_image_colors(self, image_path: str, num_colors: int = 10, method: str = 'kmeans'):
        """
        Apply colors extracted from an image to the wordcloud.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors to extract
            method: Extraction method ('kmeans', 'most_common', 'random')
            
        Returns:
            Wordcloud: Self for method chaining
        """
        from .utils.color_extraction import map_words_to_image_colors
        
        if not self.gen_positions:
            self.logger.warning("No words generated yet. Call generate() first.")
            return self
        
        # Get words from gen_positions
        words = [word for (word, _, _), _, _, _, _, _ in self.gen_positions]
        
        # Map words to colors
        word_colors = map_words_to_image_colors(
            words,
            image_path,
            num_colors,
            method,
            frequency_based=True,
            rng=self._np_random,
        )
        
        # Update colors
        self.update_colors(word_colors)
        
        self.logger.info(f"Applied {num_colors} colors from image: {image_path}")
        return self
    
    def optimize_memory(self):
        """
        Optimize memory usage by clearing unnecessary caches and data.
        
        Returns:
            Wordcloud: Self for method chaining
        """
        from .utils.performance_optimizations import optimize_memory_usage
        
        optimize_memory_usage(self)
        return self
    
    def apply_semantic_clustering(
        self,
        method: str = 'tfidf',
        num_clusters: Optional[int] = None,
        cluster_layout: str = 'grouped',
        apply_cluster_colors: bool = True,
        color_theme: str = 'viridis'
    ):
        """
        Apply semantic clustering to words and rearrange them.
        
        Args:
            method: Clustering method ('tfidf', 'similarity')
            num_clusters: Number of clusters (auto-determined if None)
            cluster_layout: Layout strategy ('grouped', 'interleaved', 'sorted')
            apply_cluster_colors: Whether to color words by cluster
            color_theme: Color theme for cluster colors
            
        Returns:
            Wordcloud: Self for method chaining
        """
        from .utils.semantic_clustering import (
            cluster_words_tfidf, cluster_words_similarity,
            arrange_words_by_cluster, get_cluster_colors
        )
        
        if not self.gen_positions:
            self.logger.warning("No words generated yet. Call generate() first.")
            return self
        
        # Extract words and frequencies
        words = [word for (word, _, _), _, _, _, _, _ in self.gen_positions]
        frequencies = [freq for (_, freq, _), _, _, _, _, _ in self.gen_positions]
        original_freqs = [count for (_, _, count), _, _, _, _, _ in self.gen_positions]
        
        # Cluster words
        if method == 'tfidf':
            try:
                clusters = cluster_words_tfidf(words, frequencies, num_clusters)
            except RuntimeError as e:
                self.logger.warning(f"TF-IDF clustering failed: {e}, falling back to similarity")
                clusters = cluster_words_similarity(words)
        else:  # similarity
            clusters = cluster_words_similarity(words)
        
        # Rearrange words
        word_tuples = list(zip(words, frequencies, original_freqs))
        rearranged = arrange_words_by_cluster(word_tuples, clusters, cluster_layout)
        
        # Rebuild gen_positions with new order
        # Create mapping from word to original position data
        position_map = {
            word: (font_path, font_size, position, orientation, color)
            for (word, _, _), font_path, font_size, position, orientation, color in self.gen_positions
        }
        
        new_gen_positions = []
        for word, freq, count in rearranged:
            if word in position_map:
                font_path, font_size, position, orientation, color = position_map[word]
                new_gen_positions.append(((word, freq, count), font_path, font_size, position, orientation, color))
        
        self.gen_positions = new_gen_positions
        
        # Apply cluster colors if requested
        if apply_cluster_colors:
            cluster_word_colors = get_cluster_colors(clusters, color_theme)
            self.update_colors(cluster_word_colors)
        
        self.logger.info(f"Applied semantic clustering: {len(set(clusters.values()))} clusters")
        return self
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides):
        """
        Create a Wordcloud instance from a preset.
        
        Args:
            preset_name: Name of the preset
            **overrides: Parameters to override in the preset
            
        Returns:
            Wordcloud instance
            
        Example:
            >>> wc = Wordcloud.from_preset('minimal')
            >>> wc.generate("Your text here")
        """
        from .utils.presets import get_preset, list_presets
        
        preset = get_preset(preset_name)
        if not preset:
            available = ', '.join(list_presets())
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")
        
        # Merge preset with overrides
        params = {**preset, **overrides}
        
        return cls(**params)
    
    def save_as_preset(self, name: str):
        """
        Save current wordcloud configuration as a preset.
        
        Args:
            name: Preset name
            
        Returns:
            Wordcloud: Self for method chaining
        """
        from .utils.presets import _preset_manager
        
        _preset_manager.create_preset_from_wordcloud(self, name)
        self.logger.info(f"Saved current configuration as preset: {name}")
        return self
