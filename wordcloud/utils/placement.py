"""
Word placement strategies used by the Wordcloud generator.

These helpers locate a top-left coordinate for a word's bounding box while
delegating collision checks to a supplied callback.
"""

from __future__ import annotations

import random
import math
from typing import Callable, Optional, Tuple, List, Dict, Protocol, Set, TYPE_CHECKING

from .logging_config import get_logger

logger = get_logger(__name__)

IsValidPosition = Callable[[int, int, int, int], bool]  # y, x, height, width -> bool
if TYPE_CHECKING:
    from .integral_image import IntegralImage


class PlacementStrategy(Protocol):
    def __call__(
        self,
        integral_image: "IntegralImage",
        free_locations: Set[Tuple[int, int]],
        width_x: int,
        height_y: int,
        size_x: int,
        size_y: int,
    ) -> Optional[Tuple[int, int]]:
        ...


_CUSTOM_STRATEGIES: Dict[str, PlacementStrategy] = {}


def register_placement_strategy(name: str, strategy: PlacementStrategy) -> None:
    if not name:
        raise ValueError("Strategy name must be a non-empty string")
    from .integral_image import STRATEGIES
    if name in STRATEGIES or name in _CUSTOM_STRATEGIES:
        raise ValueError(f"Strategy '{name}' is already registered")
    _CUSTOM_STRATEGIES[name] = strategy
    STRATEGIES.append(name)


def unregister_placement_strategy(name: str) -> None:
    from .integral_image import STRATEGIES
    if name in _CUSTOM_STRATEGIES:
        del _CUSTOM_STRATEGIES[name]
    if name in STRATEGIES:
        STRATEGIES.remove(name)


def get_placement_strategy(name: str) -> Optional[PlacementStrategy]:
    return _CUSTOM_STRATEGIES.get(name)


def list_custom_placement_strategies() -> List[str]:
    return sorted(_CUSTOM_STRATEGIES.keys())

DEFAULT_STEP = 2


def find_position_random(
    canvas_width: int,
    canvas_height: int,
    word_width: int,
    word_height: int,
    is_valid_position: IsValidPosition,
    tracer: Optional[object] = None,
    max_attempts: int = 1000,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[int, int]]:
    """
    Sample random positions until a valid spot is found or attempts are exhausted.
    """
    if word_width > canvas_width or word_height > canvas_height:
        logger.warning(f"Word size {word_width}x{word_height} exceeds canvas {canvas_width}x{canvas_height}")
        return None

    logger.debug(f"Attempting random placement for size {word_width}x{word_height} (max {max_attempts} attempts)")
    rng = rng or random
    for attempt in range(max_attempts):
        pos_x = rng.randint(0, canvas_width - word_width)
        pos_y = rng.randint(0, canvas_height - word_height)

        if tracer and getattr(tracer, "is_active", False):
            tracer.draw_point(pos_x, pos_y, color="blue")

        if is_valid_position(pos_y, pos_x, word_height, word_width):
            logger.debug(f"Random placement succeeded on attempt {attempt + 1} at ({pos_x}, {pos_y})")
            return pos_x, pos_y

    logger.debug(f"Random placement failed after {max_attempts} attempts for size {word_width}x{word_height}")
    return None


def find_position_rectangular_spiral(
    canvas_width: int,
    canvas_height: int,
    word_width: int,
    word_height: int,
    is_valid_position: IsValidPosition,
    start_x: Optional[int] = None,
    start_y: Optional[int] = None,
    step: int = DEFAULT_STEP,
    tracer: Optional[object] = None,
) -> Optional[Tuple[int, int]]:
    """
    Search outward from the canvas center following a rectangular spiral.
    """
    if word_width > canvas_width or word_height > canvas_height:
        logger.warning(f"Word size {word_width}x{word_height} exceeds canvas {canvas_width}x{canvas_height}")
        return None

    center_x = start_x if start_x is not None else (canvas_width - word_width) // 2
    center_y = start_y if start_y is not None else (canvas_height - word_height) // 2
    logger.debug(f"Starting rectangular spiral from center ({center_x}, {center_y})")

    def check(x: int, y: int) -> Optional[Tuple[int, int]]:
        if tracer and getattr(tracer, "is_active", False):
            tracer.draw_point(x, y)
        if is_valid_position(y, x, word_height, word_width):
            logger.debug(f"Rectangular spiral placement succeeded at ({x}, {y})")
            return x, y
        return None

    first = check(center_x, center_y)
    if first:
        return first

    # Spiral: right, down, left, up
    directions = [(step, 0), (0, step), (-step, 0), (0, -step)]
    seg_length = 1
    x, y = center_x, center_y

    # Limit iterations to cover the whole canvas area
    max_iters = (canvas_width // step + canvas_height // step + 4) * 2
    iters = 0

    while iters < max_iters:
        for dx, dy in directions:
            for _ in range(seg_length):
                x += dx
                y += dy
                iters += 1
                result = check(x, y)
                if result:
                    return result
            # Increase segment length after horizontal moves
            if dx != 0:
                seg_length += 1

    return None


def find_position_circular(
    canvas_width: int,
    canvas_height: int,
    word_width: int,
    word_height: int,
    is_valid_position: IsValidPosition,
    center_x: Optional[int] = None,
    center_y: Optional[int] = None,
    start_radius: int = 10,
    radius_step: int = 5,
    angle_step: float = 0.1,
    tracer: Optional[object] = None,
) -> Optional[Tuple[int, int]]:
    """
    Search outward from center following a circular spiral pattern.
    
    This creates a more organic, circular layout compared to rectangular spirals.
    """
    if word_width > canvas_width or word_height > canvas_height:
        logger.warning(f"Word size {word_width}x{word_height} exceeds canvas {canvas_width}x{canvas_height}")
        return None
    
    center_x = center_x if center_x is not None else canvas_width // 2
    center_y = center_y if center_y is not None else canvas_height // 2
    
    logger.debug(f"Starting circular spiral from center ({center_x}, {center_y})")
    
    radius = start_radius
    angle = 0.0
    max_radius = math.sqrt(canvas_width**2 + canvas_height**2) / 2
    
    while radius < max_radius:
        # Calculate position on circle
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        
        # Adjust for word dimensions (center the word)
        x -= word_width // 2
        y -= word_height // 2
        
        # Check bounds
        if 0 <= x <= canvas_width - word_width and 0 <= y <= canvas_height - word_height:
            if tracer and getattr(tracer, "is_active", False):
                tracer.draw_point(x, y, color="green")
            
            if is_valid_position(y, x, word_height, word_width):
                logger.debug(f"Circular spiral placement succeeded at ({x}, {y})")
                return x, y
        
        # Advance angle
        angle += angle_step
        # Increase radius periodically
        if angle >= 2 * math.pi:
            angle = 0.0
            radius += radius_step
    
    return None


def find_position_hierarchical(
    canvas_width: int,
    canvas_height: int,
    word_width: int,
    word_height: int,
    is_valid_position: IsValidPosition,
    word_index: int = 0,
    total_words: int = 1,
    center_x: Optional[int] = None,
    center_y: Optional[int] = None,
    tracer: Optional[object] = None,
) -> Optional[Tuple[int, int]]:
    """
    Hierarchical placement: most important words near center, others in rings.
    
    This creates a hierarchical layout where word importance (index) determines
    distance from center.
    """
    if word_width > canvas_width or word_height > canvas_height:
        logger.warning(f"Word size {word_width}x{word_height} exceeds canvas {canvas_width}x{canvas_height}")
        return None
    
    center_x = center_x if center_x is not None else canvas_width // 2
    center_y = center_y if center_y is not None else canvas_height // 2
    
    # Calculate ring number based on word index
    if total_words == 1:
        ring = 0
    else:
        ring = int((word_index / total_words) * 5)  # 5 rings max
    
    # Calculate radius for this ring
    max_radius = min(canvas_width, canvas_height) // 2 - max(word_width, word_height)
    radius = (ring / 5.0) * max_radius
    
    # Try positions in this ring
    num_positions = max(8, ring * 4)  # More positions for outer rings
    for i in range(num_positions):
        angle = (2 * math.pi * i) / num_positions
        x = int(center_x + radius * math.cos(angle) - word_width // 2)
        y = int(center_y + radius * math.sin(angle) - word_height // 2)
        
        # Check bounds
        if 0 <= x <= canvas_width - word_width and 0 <= y <= canvas_height - word_height:
            if tracer and getattr(tracer, "is_active", False):
                tracer.draw_point(x, y, color="purple")
            
            if is_valid_position(y, x, word_height, word_width):
                logger.debug(f"Hierarchical placement succeeded at ({x}, {y})")
                return x, y
    
    # Fallback to random if ring placement fails
    return find_position_random(canvas_width, canvas_height, word_width, word_height, is_valid_position)


def find_position_grid(
    canvas_width: int,
    canvas_height: int,
    word_width: int,
    word_height: int,
    is_valid_position: IsValidPosition,
    grid_cols: Optional[int] = None,
    grid_rows: Optional[int] = None,
    start_x: int = 0,
    start_y: int = 0,
    tracer: Optional[object] = None,
) -> Optional[Tuple[int, int]]:
    """
    Grid-based placement: try positions on a regular grid.
    
    This is efficient for dense wordclouds and provides structured layouts.
    """
    if word_width > canvas_width or word_height > canvas_height:
        logger.warning(f"Word size {word_width}x{word_height} exceeds canvas {canvas_width}x{canvas_height}")
        return None
    
    # Calculate grid dimensions
    if grid_cols is None:
        grid_cols = max(1, canvas_width // (word_width + 10))
    if grid_rows is None:
        grid_rows = max(1, canvas_height // (word_height + 10))
    
    col_step = canvas_width // grid_cols
    row_step = canvas_height // grid_rows
    
    logger.debug(f"Grid placement: {grid_cols}x{grid_rows} grid")
    
    # Try grid positions
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = start_x + col * col_step
            y = start_y + row * row_step
            
            # Ensure within bounds
            x = min(x, canvas_width - word_width)
            y = min(y, canvas_height - word_height)
            
            if x < 0 or y < 0:
                continue
            
            if tracer and getattr(tracer, "is_active", False):
                tracer.draw_point(x, y, color="orange")
            
            if is_valid_position(y, x, word_height, word_width):
                logger.debug(f"Grid placement succeeded at ({x}, {y})")
                return x, y
    
    return None


def find_position_force_directed(
    canvas_width: int,
    canvas_height: int,
    word_width: int,
    word_height: int,
    is_valid_position: IsValidPosition,
    placed_words: Optional[List[Tuple[int, int, int, int]]] = None,
    iterations: int = 50,
    tracer: Optional[object] = None,
) -> Optional[Tuple[int, int]]:
    """
    Force-directed placement: uses physics simulation to find optimal position.
    
    This creates more natural, evenly-spaced layouts by simulating repulsion
    between words.
    """
    if word_width > canvas_width or word_height > canvas_height:
        logger.warning(f"Word size {word_width}x{word_height} exceeds canvas {canvas_width}x{canvas_height}")
        return None
    
    placed_words = placed_words or []
    
    # Start from center
    x = canvas_width // 2 - word_width // 2
    y = canvas_height // 2 - word_height // 2
    
    # Simple force-directed: repel from existing words
    for _ in range(iterations):
        fx, fy = 0.0, 0.0
        
        for px, py, pw, ph in placed_words:
            # Calculate distance
            dx = (x + word_width // 2) - (px + pw // 2)
            dy = (y + word_height // 2) - (py + ph // 2)
            dist = math.sqrt(dx*dx + dy*dy + 1)  # +1 to avoid division by zero
            
            # Repulsion force (inverse square)
            force = 1000.0 / (dist * dist)
            fx += (dx / dist) * force
            fy += (dy / dist) * force
        
        # Apply force with damping
        x += int(fx * 0.1)
        y += int(fy * 0.1)
        
        # Keep within bounds
        x = max(0, min(x, canvas_width - word_width))
        y = max(0, min(y, canvas_height - word_height))
        
        # Check if valid position
        if is_valid_position(y, x, word_height, word_width):
            logger.debug(f"Force-directed placement succeeded at ({x}, {y})")
            return x, y
    
    # Fallback to random if force-directed fails
    return find_position_random(canvas_width, canvas_height, word_width, word_height, is_valid_position)

