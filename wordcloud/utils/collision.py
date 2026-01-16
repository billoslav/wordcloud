"""
Collision detection utilities for the wordcloud library.

This module provides collision detection algorithms and spatial data structures 
for efficient word placement in wordclouds.
"""

import logging
import math
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Generator, Callable
import numpy as np
from collections import defaultdict

from .logging_config import get_logger

logger = get_logger(__name__)

CollisionDetectorFactory = Callable[..., "CollisionDetector"]
_CUSTOM_DETECTORS: Dict[str, CollisionDetectorFactory] = {}


def register_collision_detector(name: str, factory: CollisionDetectorFactory) -> None:
    if not name:
        raise ValueError("Detector name must be a non-empty string")
    if name in _CUSTOM_DETECTORS:
        raise ValueError(f"Detector '{name}' is already registered")
    _CUSTOM_DETECTORS[name] = factory


def unregister_collision_detector(name: str) -> None:
    if name in _CUSTOM_DETECTORS:
        del _CUSTOM_DETECTORS[name]


def list_collision_detectors() -> List[str]:
    return sorted(_CUSTOM_DETECTORS.keys())


class CollisionDetector:
    """
    Base class for collision detection algorithms.
    
    This abstract class defines the interface for collision detection algorithms.
    Different implementations can optimize for specific scenarios.
    """
    
    def __init__(self, width: int, height: int, rng: Optional[random.Random] = None):
        """
        Initialize the collision detector.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
        """
        self.width = width
        self.height = height
        self._rng = rng or random
        
    def add_rectangle(self, rect: Tuple[int, int, int, int]) -> None:
        """
        Add a rectangle (representing an occupied area) to the detector.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple.
        """
        raise NotImplementedError("Subclasses must implement add_rectangle")
    
    def check_collision(self, rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if a given rectangle collides with any previously added rectangles.
        
        Args:
            rect: Rectangle to check, as (x, y, width, height) tuple.
            
        Returns:
            True if there is a collision, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement check_collision")

    def is_position_available(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Check if a position is available (no collision and within bounds).
        
        This is often the inverse of check_collision, but explicitly checks bounds.

        Args:
            x: X-coordinate of the top-left corner.
            y: Y-coordinate of the top-left corner.
            width: Width of the area to check.
            height: Height of the area to check.
            
        Returns:
            True if the position is available, False otherwise.
        """
        # Default implementation: check bounds and then collision
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            return False
        return not self.check_collision((x, y, width, height))
        
    def clear(self) -> None:
        """Clear all stored rectangles or collision state."""
        raise NotImplementedError("Subclasses must implement clear")


class BruteForceCollisionDetector(CollisionDetector):
    """
    Simple collision detector that checks each rectangle against all others.
    
    O(n) lookup time, suitable for small numbers of rectangles.
    """
    
    def __init__(self, width: int, height: int, rng: Optional[random.Random] = None):
        """
        Initialize the brute force collision detector.
        """
        super().__init__(width, height, rng=rng)
        self.rectangles: List[Tuple[int, int, int, int]] = []
        
    def add_rectangle(self, rect: Tuple[int, int, int, int]) -> None:
        """
        Add a rectangle to the list.
        """
        if len(rect) != 4:
            raise ValueError("Rectangle must be a (x, y, width, height) tuple")
        self.rectangles.append(rect)
    
    def check_collision(self, rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if `rect` overlaps with any stored rectangle.
        Does not check canvas bounds here, assuming `is_position_available` handles it.
        """
        rect_x, rect_y, rect_w, rect_h = rect
        
        for r in self.rectangles:
            r_x, r_y, r_w, r_h = r
            # Check for overlap using Separating Axis Theorem logic (inverted)
            if (rect_x < r_x + r_w and rect_x + rect_w > r_x and
                rect_y < r_y + r_h and rect_y + rect_h > r_y):
                return True  # Collision detected
        
        return False  # No collision

    def clear(self) -> None:
        """Clear all stored rectangles."""
        self.rectangles = []


class GridCollisionSystem(CollisionDetector):
    """
    Grid-based collision detection using a sparse grid of occupied cells.
    
    Divides the canvas into a grid. A cell is marked as occupied if any part
    of a placed rectangle overlaps it. Collision checking involves checking 
    only the cells overlapped by the new rectangle.
    
    Provides O(1) average time complexity for collision checks, assuming 
    uniform distribution and reasonable cell size.
    
    Attributes:
        cell_size (int): Size of each square grid cell in pixels.
        grid_width (int): Number of cells horizontally.
        grid_height (int): Number of cells vertically.
        grid (defaultdict): Sparse dictionary mapping (grid_x, grid_y) cell 
                            coordinates to occupancy state (True = occupied).
    """
    
    def __init__(self, width: int, height: int, cell_size: int = 10, rng: Optional[random.Random] = None):
        """
        Initialize the grid-based collision system.
        
        Args:
            width: Width of the canvas in pixels.
            height: Height of the canvas in pixels.
            cell_size: Size of each grid cell. Smaller gives more precision but 
                       uses more memory and potentially more checks per rectangle.
        """
        super().__init__(width, height, rng=rng)
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = cell_size
        
        # Calculate grid dimensions
        self.grid_width = math.ceil(width / cell_size)
        self.grid_height = math.ceil(height / cell_size)
        
        # Initialize empty grid using defaultdict for sparsity
        self.grid = defaultdict(bool)
        logger.info(f"Initialized GridCollisionSystem ({self.grid_width}x{self.grid_height} grid, cell size {self.cell_size})")

    def _get_grid_cells(self, x: int, y: int, width: int, height: int) -> Generator[Tuple[int, int], None, None]:
        """
        Yield all grid cell indices (col, row) that overlap with a given rectangle.
        
        Args:
            x, y: Top-left coordinates of the rectangle.
            width, height: Dimensions of the rectangle.
            
        Yields:
            Tuple of (grid_x, grid_y) for each overlapping cell.
        """
        start_col = max(0, x // self.cell_size)
        start_row = max(0, y // self.cell_size)
        end_col = min(self.grid_width, (x + width + self.cell_size - 1) // self.cell_size)
        end_row = min(self.grid_height, (y + height + self.cell_size - 1) // self.cell_size)
        
        for grid_y in range(start_row, end_row):
            for grid_x in range(start_col, end_col):
                yield grid_x, grid_y

    def add_rectangle(self, rect: Tuple[int, int, int, int]) -> None:
        """
        Mark the grid cells overlapped by the rectangle as occupied.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple.
        """
        x, y, width, height = rect
        for grid_x, grid_y in self._get_grid_cells(x, y, width, height):
            self.grid[(grid_x, grid_y)] = True
            
    def check_collision(self, rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if the rectangle collides with any occupied grid cells.
        
        Args:
            rect: Rectangle to check, as (x, y, width, height) tuple.
            
        Returns:
            True if any grid cell overlapped by `rect` is marked as occupied, 
            False otherwise.
        """
        x, y, width, height = rect
        for grid_x, grid_y in self._get_grid_cells(x, y, width, height):
            if self.grid[(grid_x, grid_y)]:
                return True
        return False

    def is_position_available(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Check if a rectangular area is within bounds and does not collide with 
        occupied grid cells.
        
        Args:
            x, y: Top-left coordinates of the area.
            width, height: Dimensions of the area.
            
        Returns:
            True if the position is available, False otherwise.
        """
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            return False
        return not self.check_collision((x, y, width, height))
    
    def clear(self) -> None:
        """Clear the grid, removing all occupied cells."""
        self.grid.clear()
        logger.debug("GridCollisionSystem cleared.")

    def get_random_available_position(self, width: int, height: int, max_attempts: int = 100) -> Optional[Tuple[int, int]]:
        """
        Attempt to find a random available top-left position for a given rectangle size.
        
        Args:
            width: Width of the rectangle to place.
            height: Height of the rectangle to place.
            max_attempts: Number of random positions to try before systematic search.
            
        Returns:
            A tuple (x, y) of an available position, or None if none found.
        """
        if width > self.width or height > self.height: 
            return None
            
        for _ in range(max_attempts):
            x = self._rng.randint(0, self.width - width)
            y = self._rng.randint(0, self.height - height)
            
            if self.is_position_available(x, y, width, height):
                return x, y
        
        logger.debug(f"Random position search failed after {max_attempts} attempts, falling back to systematic search.")
        return self.find_first_available_position(width, height)
    
    def find_first_available_position(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """
        Find the first available top-left position for a rectangle using a 
        systematic search.
        
        Args:
            width: Width of the rectangle to place.
            height: Height of the rectangle to place.
            
        Returns:
            Tuple (x, y) of the first available position found, or None otherwise.
        """
        if width > self.width or height > self.height: 
            return None
            
        for y in range(0, self.height - height + 1, self.cell_size):
            for x in range(0, self.width - width + 1, self.cell_size):
                if self.is_position_available(x, y, width, height):
                    return x, y
        
        logger.debug("Grid-aligned search failed, starting fine-grained search.")
        for y in range(0, self.height - height + 1):
            for x in range(0, self.width - width + 1):
                if self.is_position_available(x, y, width, height):
                    return x, y
        
        logger.debug(f"Systematic search failed to find position for size ({width}x{height}).")
        return None


class QuadtreeNode:
    """
    Node in a quadtree spatial data structure.
    
    Quadtrees recursively subdivide space into four quadrants, allowing for
    efficient spatial queries.
    """
    
    def __init__(
        self, 
        x: int, 
        y: int, 
        width: int, 
        height: int, 
        max_depth: int = 8,
        max_objects: int = 10,
        depth: int = 0
    ):
        """
        Initialize a quadtree node.
        
        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of this node's region
            height: Height of this node's region
            max_depth: Maximum depth of the quadtree
            max_objects: Maximum number of objects before splitting
            depth: Current depth of this node
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.max_objects = max_objects
        self.depth = depth
        
        self.objects: List[Tuple[int, int, int, int]] = []
        self.children: List['QuadtreeNode'] = []
        self.is_leaf = True
    
    def clear(self) -> None:
        """Clear all objects and reset children."""
        self.objects = []
        self.children = []
        self.is_leaf = True
    
    def contains(self, rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if this node fully contains a rectangle.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple
            
        Returns:
            True if the rectangle is fully contained in this node
        """
        rect_x, rect_y, rect_w, rect_h = rect
        return (self.x <= rect_x and rect_x + rect_w <= self.x + self.width and
                self.y <= rect_y and rect_y + rect_h <= self.y + self.height)
    
    def intersects(self, rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if this node intersects a rectangle.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple
            
        Returns:
            True if the rectangle intersects this node
        """
        rect_x, rect_y, rect_w, rect_h = rect
        return not (rect_x > self.x + self.width or rect_x + rect_w < self.x or
                   rect_y > self.y + self.height or rect_y + rect_h < self.y)
    
    def split(self) -> None:
        """Split this node into four children."""
        if not self.is_leaf:
            return
        
        half_width = self.width // 2
        half_height = self.height // 2
        new_depth = self.depth + 1
        
        self.children = [
            QuadtreeNode(self.x, self.y, half_width, half_height, 
                        self.max_depth, self.max_objects, new_depth),
            QuadtreeNode(self.x + half_width, self.y, half_width, half_height, 
                        self.max_depth, self.max_objects, new_depth),
            QuadtreeNode(self.x, self.y + half_height, half_width, half_height, 
                        self.max_depth, self.max_objects, new_depth),
            QuadtreeNode(self.x + half_width, self.y + half_height, half_width, half_height, 
                        self.max_depth, self.max_objects, new_depth)
        ]
        
        for obj in self.objects:
            for child in self.children:
                if child.contains(obj):
                    child.insert(obj)
                    break
        
        self.is_leaf = False
    
    def insert(self, rect: Tuple[int, int, int, int]) -> None:
        """
        Insert a rectangle into the quadtree.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple
        """
        if not self.is_leaf:
            for child in self.children:
                if child.contains(rect):
                    child.insert(rect)
                    return
            self.objects.append(rect)
            return
        
        self.objects.append(rect)
        
        if self.is_leaf and len(self.objects) > self.max_objects and self.depth < self.max_depth:
            self.split()
    
    def query_range(self, rect: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Find all rectangles that intersect with the given rectangle.
        
        Args:
            rect: Query rectangle as (x, y, width, height) tuple
            
        Returns:
            List of intersecting rectangles
        """
        result = []
        
        if not self.intersects(rect):
            return result
        
        for obj in self.objects:
            if rectangles_overlap(rect, obj):
                result.append(obj)
        
        if self.is_leaf:
            return result
        
        for child in self.children:
            result.extend(child.query_range(rect))
        
        return result


class QuadtreeCollisionDetector(CollisionDetector):
    """
    Quadtree-based collision detector for very efficient collision checking.
    
    This implementation uses a quadtree spatial data structure to organize
    rectangles, allowing for logarithmic time complexity in the average case.
    """
    
    def __init__(self, width: int, height: int, max_depth: int = 8, max_objects: int = 10, rng: Optional[random.Random] = None):
        """
        Initialize the quadtree collision detector.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
            max_depth: Maximum depth of the quadtree
            max_objects: Maximum number of objects per quadtree node
        """
        super().__init__(width, height, rng=rng)
        self.root = QuadtreeNode(0, 0, width, height, max_depth, max_objects)
        
    def add_rectangle(self, rect: Tuple[int, int, int, int]) -> None:
        """
        Add a rectangle to the collision detector.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple
        """
        self.root.insert(rect)
    
    def check_collision(self, rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if a rectangle collides with any existing rectangles.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple
            
        Returns:
            True if there is a collision, False otherwise
        """
        rect_x, rect_y, rect_w, rect_h = rect
        
        if rect_x < 0 or rect_y < 0 or rect_x + rect_w > self.width or rect_y + rect_h > self.height:
            return True
        
        potential_collisions = self.root.query_range(rect)
        
        for r in potential_collisions:
            r_x, r_y, r_w, r_h = r
            if (rect_x < r_x + r_w and rect_x + rect_w > r_x and
                rect_y < r_y + r_h and rect_y + rect_h > r_y):
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear all stored rectangles."""
        self.root.clear()


class MaskCollisionDetector(CollisionDetector):
    """
    Mask-based collision detector using a boolean array.
    
    This implementation uses a 2D boolean array to represent occupied pixels,
    which can be very efficient for certain types of wordclouds, especially
    with irregular shapes.
    """
    
    def __init__(self, width: int, height: int, mask: Optional[np.ndarray] = None, rng: Optional[random.Random] = None):
        """
        Initialize the mask collision detector.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
            mask: Optional initial mask as 2D boolean array (True = occupied)
        """
        super().__init__(width, height, rng=rng)
        
        if mask is not None:
            if mask.shape[0] != height or mask.shape[1] != width:
                logger.warning(f"Mask dimensions {mask.shape} don't match canvas {width}x{height}, resizing")
                self.mask = np.zeros((height, width), dtype=bool)
                h = min(height, mask.shape[0])
                w = min(width, mask.shape[1])
                self.mask[:h, :w] = mask[:h, :w]
            else:
                self.mask = mask.copy()
        else:
            self.mask = np.zeros((height, width), dtype=bool)
        
    def add_rectangle(self, rect: Tuple[int, int, int, int]) -> None:
        """
        Add a rectangle to the collision detector.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple
        """
        rect_x, rect_y, rect_w, rect_h = rect
        
        x1 = max(0, rect_x)
        y1 = max(0, rect_y)
        x2 = min(self.width, rect_x + rect_w)
        y2 = min(self.height, rect_y + rect_h)
        
        self.mask[y1:y2, x1:x2] = True
    
    def check_collision(self, rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if a rectangle collides with any occupied pixels.
        
        Args:
            rect: Rectangle as (x, y, width, height) tuple
            
        Returns:
            True if there is a collision, False otherwise
        """
        rect_x, rect_y, rect_w, rect_h = rect
        
        if rect_x < 0 or rect_y < 0 or rect_x + rect_w > self.width or rect_y + rect_h > self.height:
            return True
        
        return np.any(self.mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w])
    
    def clear(self) -> None:
        """Clear all occupied pixels."""
        self.mask.fill(False)
    
    def set_mask(self, mask: np.ndarray) -> None:
        """
        Set a new mask (where True values are occupied).
        
        Args:
            mask: 2D boolean array where True = occupied
        """
        if mask.shape[0] != self.height or mask.shape[1] != self.width:
            logger.warning(f"Mask dimensions {mask.shape} don't match canvas {self.width}x{self.height}, resizing")
            new_mask = np.zeros((self.height, self.width), dtype=bool)
            h = min(self.height, mask.shape[0])
            w = min(self.width, mask.shape[1])
            new_mask[:h, :w] = mask[:h, :w]
            self.mask = new_mask
        else:
            self.mask = mask.copy()


def rectangles_overlap(rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
    """
    Check if two rectangles overlap.
    
    Args:
        rect1: First rectangle as (x, y, width, height) tuple
        rect2: Second rectangle as (x, y, width, height) tuple
        
    Returns:
        True if the rectangles overlap, False otherwise
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def point_in_rectangle(point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
    """
    Check if a point is inside a rectangle.
    
    Args:
        point: Point as (x, y) tuple
        rect: Rectangle as (x, y, width, height) tuple
        
    Returns:
        True if the point is inside the rectangle, False otherwise
    """
    px, py = point
    rx, ry, rw, rh = rect
    
    return rx <= px < rx + rw and ry <= py < ry + rh


def create_collision_detector(
    detector_type: str,
    width: int,
    height: int,
    mask: Optional[np.ndarray] = None,
    rng: Optional[random.Random] = None,
    **kwargs
) -> CollisionDetector:
    """
    Factory function to create different types of collision detectors.

    Args:
        detector_type: The type of detector to create ('brute', 'grid', 'quadtree', 'mask').
        width: Canvas width.
        height: Canvas height.
        mask: Optional mask array (required for 'mask' type).
        **kwargs: Additional arguments for specific detector types.

    Returns:
        An instance of a CollisionDetector subclass.
    
    Raises:
        ValueError: If the detector_type is unknown or required arguments are missing.
    """
    detector_type = detector_type.lower()
    logger.info(f"Creating collision detector: type={detector_type}, size=({width}x{height})")

    if detector_type in _CUSTOM_DETECTORS:
        return _CUSTOM_DETECTORS[detector_type](width=width, height=height, mask=mask, rng=rng, **kwargs)

    if detector_type == 'brute':
        return BruteForceCollisionDetector(width, height, rng=rng)
    elif detector_type == 'grid':
        cell_size = kwargs.get('cell_size', 20)
        return GridCollisionSystem(width, height, cell_size=cell_size, rng=rng)
    elif detector_type == 'quadtree':
        max_depth = kwargs.get('max_depth', 8)
        max_objects = kwargs.get('max_objects', 10)
        return QuadtreeCollisionDetector(width, height, max_depth=max_depth, max_objects=max_objects, rng=rng)
    elif detector_type == 'mask':
        if mask is None:
            raise ValueError("Mask array must be provided for 'mask' collision detector.")
        return MaskCollisionDetector(width, height, mask=mask, rng=rng)
    else:
        raise ValueError(f"Unknown collision detector type: {detector_type}")

