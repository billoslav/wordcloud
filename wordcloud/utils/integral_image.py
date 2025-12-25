import numpy as np
from random import randint
from PIL import Image, ImageDraw
import os
import shutil
from scipy import spatial
import quads
import math
import time

from .logging_config import get_logger

logger = get_logger(__name__)

# Define available placement strategies
STRATEGIES = ["random", "brute", "archimedian", "rectangular", "archimedian_reverse", "rectangular_reverse", "KDTree", "quad", "pytag", "pytag_reverse"]

class IntegralImage:
    """
    Implements an integral image for efficient word placement and collision detection.
    
    The integral image is a data structure that allows for fast area sum calculations,
    which is essential for determining if a region is available for placing a word.
    It also implements various placement strategies for positioning words in the wordcloud.
    
    Attributes:
        height (int): Height of the integral image in pixels
        width (int): Width of the integral image in pixels
        integral (numpy.ndarray): 2D array representing the integral image
        tracing (bool): Whether to generate tracing images for debugging
        trace_margin (int): Margin size for tracing images
        half_trace_margin (float): Half of the trace margin for calculations
        structure_created (bool): Flag indicating if the tracing directory structure exists
        directory_name (str): Directory for storing tracing images
        INCREASE (int): Step size increment for spiral strategies
        DEFAULT_STEP (int): Default step size for movement in placement strategies
    """
    def __init__(self, height, width, tracing=False, mask=None):
        """
        Initialize an IntegralImage instance.
        
        Args:
            height (int): Height of the integral image in pixels
            width (int): Width of the integral image in pixels
            tracing (bool, optional): Whether to generate tracing images for debugging.
                Defaults to False.
        """
        self.height = height
        self.width = width
        
        self.INCREASE = 5
        self.DEFAULT_STEP = 2
        
        self.tracing = tracing
        self.trace_margin = 200
        self.half_trace_margin = self.trace_margin / 2
        self.structure_created = False
        self.directory_name = os.getcwd() + "/Tracking"
        self.mask = mask  # Optional mask processor with is_position_available

        # We are not using any mask, so the initial Image is filled with zeros
        self.integral = np.zeros((height, width), dtype=np.uint64)
        
        logger.info(f"IntegralImage initialized: {width}x{height}, tracing={tracing}, mask={'provided' if mask else 'none'}")
        logger.debug(f"Integral image array shape: {self.integral.shape}")
                
    def find_position(self, size_x, size_y, place_strategy="random", word_to_write=""):
        """
        Find a suitable position for a word using the specified strategy.
        
        This method searches for an available space in the integral image where a word
        with the given dimensions can be placed without overlapping existing words.
        
        Args:
            size_x (int): Width of the word's bounding box
            size_y (int): Height of the word's bounding box
            place_strategy (str, optional): Strategy to use for word placement.
                Must be one of the strategies defined in STRATEGIES.
                Defaults to "random".
            word_to_write (str, optional): The word being placed, used for tracing.
                Defaults to empty string.
                
        Returns:
            tuple or None: (x, y) coordinates for the word position if found, None otherwise
            
        Raises:
            ValueError: If size_x or size_y is negative, or if place_strategy is invalid
        """
        if size_y < 0 or size_x < 0:
            raise ValueError("Negative size of the image!")
        
        height = self.height
        width = self.width

        # Calculate initial center position
        height_y = (height - size_y) // 2
        width_x = (width - size_x) // 2

        free_locations = []

        # Setup tracing if enabled
        if self.tracing: self.tracing_setup(size_x, size_y, place_strategy, word_to_write)

        # Find all free locations
        for line_y in range(height - size_y):
            for line_x in range(width - size_x):
            
                if self.is_valid_position(line_y, line_x, size_y, size_x):
                    
                    # If we check Brute force here, it gets aprox. 40% faster
                    if place_strategy == "brute":
                        return (line_x, line_y)
                    
                    free_locations.append((line_x, line_y))

        # If we cannot find any location, return None
        if not free_locations:
            logger.debug(f"No free locations found for size {size_x}x{size_y}")
            return None

        if place_strategy not in STRATEGIES:
            logger.error(f"Invalid placement strategy: '{place_strategy}'. Available: {STRATEGIES}")
            raise ValueError(f"Incorrect placing strategy! The '{place_strategy}' is not defined.")

        # Convert to set for O(1) lookups in spiral algorithms
        free_locations_set = set(free_locations)
        
        logger.debug(f"Found {len(free_locations)} free locations, using strategy '{place_strategy}'")
        method_to_call = getattr(self, place_strategy, None)
        if method_to_call is None:
            logger.error(f"Strategy '{place_strategy}' is not implemented")
            raise ValueError(f"Strategy '{place_strategy}' is not implemented.")
        result = method_to_call(free_locations_set, width_x, height_y, size_x, size_y)
        if result:
            logger.debug(f"Position found using strategy '{place_strategy}': {result}")
        else:
            logger.debug(f"Strategy '{place_strategy}' did not find a position")
        return result

    def is_valid_position(self, pos_y, pos_x, size_y, size_x):
        """
        Check if a position is available for placing a word.
        
        Uses the integral image to efficiently determine if the rectangular area
        at the given position is empty (contains no other words).
        
        Args:
            pos_y (int): Y-coordinate of top-left corner
            pos_x (int): X-coordinate of top-left corner
            size_y (int): Height of the area to check
            size_x (int): Width of the area to check
            
        Returns:
            bool: True if the position is valid (empty), False otherwise
            
        Raises:
            ValueError: If any coordinate or size is negative
        """
        if pos_y < 0 or pos_x < 0 or size_y < 0 or size_x < 0:
            raise ValueError("Negative size or coordinates of the image!")

        if self.mask is not None and not self.mask.is_position_available(pos_y, pos_x, size_y, size_x):
            return False
        
        area = self.integral[pos_y, pos_x] + self.integral[pos_y + size_y, pos_x + size_x]
        area -= self.integral[pos_y + size_y, pos_x] + self.integral[pos_y, pos_x + size_x]
            
        return not area
        
    def check_bounds(self, x, y, size_x, size_y):
        """
        Check if a position is out of bounds or requires special handling.
        
        This method is used by the placement strategies to determine if a position
        is completely outside the valid area or if it requires special handling.
        
        Args:
            x (int): X-coordinate to check
            y (int): Y-coordinate to check
            size_x (int): Width of the area
            size_y (int): Height of the area
            
        Returns:
            bool: True if position requires special handling (multiple edges violated),
                 False if the position is within bounds or only one edge is violated
        """
        return sum([x > (self.width - size_x), y > (self.height - size_y), y < 0, x < 0]) >= 2
        
    def draw_trace_point(self, pos_x, pos_y):
        """
        Draw a point on the tracking image for visualization.
        
        Used during debugging to visualize the path taken by placement strategies.
        Only has an effect if tracing is enabled.
        
        Args:
            pos_x (int): X-coordinate of the point to draw
            pos_y (int): Y-coordinate of the point to draw
        """
        if self.tracing:
            self.track_draw.point([(pos_x + self.half_trace_margin, pos_y + self.half_trace_margin)], fill="red")

    def save_trace_img(self):
        """
        Save the current tracking image to disk.
        
        Only has an effect if tracing is enabled. The image shows the path
        taken by the placement strategy when finding a position for a word.
        """
        if self.tracing:
            self.tracking_img.save(self.trace_img_name)


    
    # Define placement strategies as inner functions
    def random(self, free_locations, width_x, height_y, size_x, size_y):
        """
        Random placement strategy - selects a random position from available locations.
        
        This is the simplest placement strategy, offering good performance but less
        visually appealing arrangements compared to other strategies.
        
        Args:
            free_locations (set): Set of available (x, y) coordinate tuples
            width_x (int): X-coordinate of the center point (used by other strategies)
            height_y (int): Y-coordinate of the center point (used by other strategies)
            size_x (int): Width of the word bounding box (used by other strategies)
            size_y (int): Height of the word bounding box (used by other strategies)
            
        Returns:
            tuple: Selected (x, y) position for word placement
        """
        # Convert to tuple for random selection (set doesn't support indexing)
        locations_tuple = tuple(free_locations)
        return locations_tuple[randint(0, len(locations_tuple) - 1)]

    def rectangular_code(self, free_locations, width_x, height_y, size_x, size_y, reverse=False):
        """
        Implementation of rectangular spiral placement strategy.
        
        This algorithm tries to place words in a rectangular spiral pattern,
        either from the center outward (forward) or from the outside inward (reverse).
        
        Args:
            free_locations (list): List of available (x, y) coordinate tuples
            width_x (int): Initial X-coordinate for spiral center
            height_y (int): Initial Y-coordinate for spiral center
            size_x (int): Width of the word bounding box
            size_y (int): Height of the word bounding box
            reverse (bool, optional): If True, use reverse spiral (outside-in).
                Defaults to False (inside-out).
                
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no position found
        """
        max_width = self.width - size_x if reverse else self.width
        max_height = self.height - size_y if reverse else self.height
        direction = 0

        for n in range(max(self.width, self.height)):
            self.draw_trace_point(width_x, height_y)
                
            if (width_x, height_y) in free_locations:
                self.save_trace_img()
                return width_x, height_y
                
            if self.check_bounds(width_x, height_y, size_x, size_y):
                break

            direction = n % 4
            axis = n % 2

            where_to_next = [
                (max_width - width_x - self.INCREASE, height_y, self.DEFAULT_STEP),  # right
                (width_x, max_height - height_y - self.INCREASE, self.DEFAULT_STEP),  # down
                (max_width - width_x + self.INCREASE, height_y, -self.DEFAULT_STEP),  # left
                (width_x, max_height - height_y + self.INCREASE, -self.DEFAULT_STEP)  # up
            ] if reverse else [
                (width_x + self.INCREASE + n, height_y, self.DEFAULT_STEP),  # right
                (width_x, height_y + self.INCREASE + n, self.DEFAULT_STEP),  # up
                (width_x - self.INCREASE - n, height_y, -self.DEFAULT_STEP),  # left
                (width_x, height_y - self.INCREASE - n, -self.DEFAULT_STEP)  # down
            ]

            end_x, end_y, defined_step = where_to_next[direction]
            start_point, stop_point = (width_x, end_x) if (width_x != end_x) else (height_y, end_y)

            # Stop if boundaries cross in reverse mode to prevent outward expansion
            if reverse:
                if (defined_step > 0 and stop_point <= start_point) or \
                   (defined_step < 0 and stop_point >= start_point):
                    break

            for current_position in range(start_point, stop_point, defined_step):
                position_x, position_y = (current_position, height_y) if (axis == 0) else (width_x, current_position)
                self.draw_trace_point(position_x, position_y)
                
                if (position_x, position_y) in free_locations:
                    self.save_trace_img()
                    return position_x, position_y

            width_x, height_y = end_x, end_y

        self.save_trace_img()
        return None

    rectangular = lambda self, free_locations, width_x, height_y, size_x, size_y: self.rectangular_code(free_locations, width_x, height_y, size_x, size_y, reverse=False)
    """Forward rectangular spiral placement strategy (center outward).
    
    A convenience lambda function that calls rectangular_code with reverse=False.
    See rectangular_code for detailed documentation.
    """
    
    rectangular_reverse = lambda self, free_locations, width_x, height_y, size_x, size_y: self.rectangular_code(free_locations, 0, 0, size_x, size_y, reverse=True)
    """Reverse rectangular spiral placement strategy (outside inward).
    
    A convenience lambda function that calls rectangular_code with reverse=True.
    See rectangular_code for detailed documentation.
    """

    def archimedian(self, free_locations, width_x, height_y, size_x, size_y):
        """
        Archimedean spiral placement strategy (center outward).
        
        This algorithm tries to place words along an Archimedean spiral pattern
        starting from the center and spiraling outward. Creates a more naturally
        curved arrangement compared to rectangular spirals.
        
        Args:
            free_locations (list): List of available (x, y) coordinate tuples
            width_x (int): X-coordinate of spiral center
            height_y (int): Y-coordinate of spiral center
            size_x (int): Width of the word bounding box
            size_y (int): Height of the word bounding box
            
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no position found
        """
        e = self.width / self.height
        for n in range(self.height * self.width):
            self.draw_trace_point(width_x, height_y)
            
            if self.check_bounds(width_x, height_y, size_x, size_y):
                break
                
            if (width_x, height_y) in free_locations:
                self.save_trace_img()
                return width_x, height_y

            width_x = width_x + int(e * (n * .1) * np.cos(n))
            height_y = height_y + int((n * .1) * np.sin(n))
                
        self.save_trace_img()
        return None

    def archimedian_reverse(self, free_locations, width_x, height_y, size_x, size_y):
        """
        Reverse Archimedean spiral placement strategy (outside inward).
        
        This algorithm tries to place words along an Archimedean spiral pattern
        starting from the outside and spiraling inward. Creates a more naturally
        curved arrangement compared to rectangular spirals.
        
        Args:
            free_locations (list): List of available (x, y) coordinate tuples
            width_x (int): Width of the wordcloud (used for calculations)
            height_y (int): Height of the wordcloud (used for calculations)
            size_x (int): Width of the word bounding box
            size_y (int): Height of the word bounding box
            
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no position found
        """
        spacing = 0.5  # Distance between turns of the spiral.
        density = 0.05  # Density of points along the spiral.

        max_radius = math.sqrt(width_x ** 2 + height_y ** 2)

        # Set up Archimedean spiral parameters
        a = 0  # Start at the center
        b = spacing  # Determines the spacing of each spiral turn

        # Calculate points along the spiral from the outside in
        theta = max_radius / b  # Start from the outer edge

        while theta > 0:
            # Calculate the radius for the current angle
            r = a + b * theta

            # Convert polar coordinates to Cartesian coordinates
            x = int(width_x + r * math.cos(theta))
            y = int(height_y + r * math.sin(theta))

            self.draw_trace_point(x, y)
            if (x, y) in free_locations:
                self.save_trace_img()
                return x, y

            # Decrease theta to move inward along the spiral
            theta -= density

        self.save_trace_img()
        return None

    def KDTree(self, free_locations, width_x, height_y, size_x, size_y):
        """
        K-D Tree placement strategy for finding the nearest available position.
        
        Uses a K-D Tree spatial data structure to efficiently find the position
        closest to the center point. This strategy is typically faster than
        spiral strategies while providing visually pleasing results.
        
        Args:
            free_locations (set): Set of available (x, y) coordinate tuples
            width_x (int): X-coordinate of target point (usually center)
            height_y (int): Y-coordinate of target point (usually center)
            size_x (int): Width of the word bounding box (not used in this strategy)
            size_y (int): Height of the word bounding box (not used in this strategy)
            
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no positions available
        """
        if not free_locations: 
            return None
        
        # Convert to list for KDTree (requires indexable sequence)
        locations_list = list(free_locations)
        (x, y) = locations_list[spatial.KDTree(locations_list).query([width_x, height_y])[1]]

        if (x, y):
            return x, y
        else: 
            return None
            
    def quad(self, free_locations, width_x, height_y, size_x, size_y):
        """
        Quad Tree placement strategy for finding the nearest available position.
        
        Uses a Quad Tree spatial data structure to efficiently find the position
        closest to the center point. Similar to KDTree but with a different
        spatial partitioning approach.
        
        Args:
            free_locations (set): Set of available (x, y) coordinate tuples
            width_x (int): X-coordinate of target point (usually center)
            height_y (int): Y-coordinate of target point (usually center)
            size_x (int): Width of the word bounding box (not used in this strategy)
            size_y (int): Height of the word bounding box (not used in this strategy)
            
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no positions available
        """
        if not free_locations: 
            return None
        
        tree = quads.QuadTree((width_x, height_y), self.width, self.height)
        
        # Set iteration works directly
        for n in free_locations:
            tree.insert(n)

        point = tree.nearest_neighbors((width_x, height_y), count=1)
                
        if not point:
            return None
        
        return point[0].x, point[0].y
        
    def pytag_code(self, free_locations, width_x, height_y, size_x, size_y, is_reverse=False):
        """
        PyTagCloud-inspired spiral placement strategy.
        
        Based on the placement strategy from the PyTagCloud project, this algorithm
        creates a spiral pattern that can run in either direction.
        
        Args:
            free_locations (list): List of available (x, y) coordinate tuples
            width_x (int): X-coordinate of spiral center
            height_y (int): Y-coordinate of spiral center
            size_x (int): Width of the word bounding box
            size_y (int): Height of the word bounding box
            is_reverse (bool, optional): Whether to reverse the spiral direction.
                Defaults to False.
                
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no position found
            
        References:
            https://github.com/atizo/PyTagCloud
        """
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            
        if is_reverse:
            directions.reverse()

        direction = directions[0]
        spl = 1

        while spl <= max(self.height, self.width):
            for step in range(spl * 2):
                if step == spl:
                    direction = directions[(spl - 1) % 4]

                width_x += direction[0] * self.DEFAULT_STEP
                height_y += direction[1] * self.DEFAULT_STEP
                self.draw_trace_point(width_x, height_y)

                if (width_x, height_y) in free_locations:
                    self.save_trace_img()
                    return width_x, height_y

            spl += 1

        self.save_trace_img()
        return None

    pytag = lambda self, free_locations, width_x, height_y, size_x, size_y: self.pytag_code(free_locations, width_x, height_y, size_x, size_y, is_reverse=False)
    """Forward PyTagCloud spiral placement strategy.
    
    A convenience lambda function that calls pytag_code with is_reverse=False.
    See pytag_code for detailed documentation.
    """
    
    pytag_reverse = lambda self, free_locations, width_x, height_y, size_x, size_y: self.pytag_code(free_locations, width_x, height_y, size_x, size_y, is_reverse=True)
    """Reverse PyTagCloud spiral placement strategy.
    
    A convenience lambda function that calls pytag_code with is_reverse=True.
    See pytag_code for detailed documentation.
    """
                
    #TODO - check Image size matching the origin size
    def update(self, new_img, x, y):
        """
        Update the integral image with a new image at the specified position.
        
        This method efficiently updates the integral image by calculating the
        cumulative sum for the new image section and incorporating it into
        the existing integral image.
        
        Args:
            new_img (numpy.ndarray): Image array to add to the integral image
            x (int): X-coordinate where to place the top-left corner of new_img
            y (int): Y-coordinate where to place the top-left corner of new_img
            
        Raises:
            ValueError: If x or y is negative
            
        Notes:
            - If the new image exceeds the boundaries of the integral image,
              it will be cropped to fit.
            - This method assumes new_img is a 2D numpy array (grayscale image).
              Color images will raise a ValueError.
        """
        if (x or y) < 0:
            raise ValueError(f"Negative coordinates not allowed: x={x}, y={y}")
        
        if x > self.width or y > self.height:
            return
        
        # Use vectorized operations for faster calculation
        recomputed = np.cumsum(np.cumsum(new_img[y:, x:],axis=1), axis=0)

        if y > 0:
            if x > 0:
                recomputed += (self.integral[y - 1, x:] - self.integral[y - 1, x - 1])
            else:
                recomputed += self.integral[y - 1, x:]
        if x > 0:
            recomputed += self.integral[y:, x - 1][:, np.newaxis]

        self.integral[y:, x:] = recomputed

    def create_folder(self, folder_name, parent_name=""):
        """
        Create a directory if it doesn't exist.
        
        Creates a folder at the specified path, handling potential errors gracefully.
        This method is used for creating folders for tracing images and results.
        
        Args:
            folder_name (str): Name of the folder to create
            parent_name (str, optional): Parent directory where folder should be created.
                Defaults to empty string (create in current directory).
                
        Notes:
            - Uses os.makedirs with exist_ok=True to avoid race conditions
            - Logs status messages about folder creation or errors
        """
        path = os.path.join(parent_name, folder_name) if parent_name else folder_name
        try:
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Directory '{path}' created/already exists.")
        except OSError as e:
            logger.error(f"Error creating directory '{path}': {e}")

    def create_tracking_structure(self, directory, place_strategy):
        """
        Create the directory structure for tracing images.
        
        Sets up the necessary directories for storing tracing images,
        organized by strategy name.
        
        Args:
            directory (str): Base directory for tracking images
            place_strategy (str): Name of the placement strategy
            
        Returns:
            str: Path to the directory where tracing images will be stored
        """
        self.create_folder(directory)
        self.create_folder(place_strategy, directory)
        
        return f"{directory}/{place_strategy}/"
    
    def tracing_setup(self, size_x, size_y, place_strategy, word_to_write):
        """
        Set up the tracing environment for visualizing placement strategies.
        
        Creates a new tracing image and prepares it for recording the path taken
        by the placement strategy. This is useful for debugging and understanding
        how different strategies work.
        
        Args:
            size_x (int): Width of the word's bounding box
            size_y (int): Height of the word's bounding box
            place_strategy (str): Name of the placement strategy being used
            word_to_write (str): Word being placed
            
        Notes:
            - The tracing image shows the boundaries of the integral image
            - Red lines indicate the word size constraints
            - Red dots show the path taken by the placement strategy
        """
        if self.structure_created:
                tracking_path = f"{self.directory_name}/{place_strategy}/"
        else:
            tracking_path = self.create_tracking_structure(self.directory_name, place_strategy)
            self.structure_created = True

        self.tracking_img = Image.new("L", (self.width + self.trace_margin, self.height + self.trace_margin), color="white")
        self.track_draw = ImageDraw.Draw(self.tracking_img)

        self.track_draw.rectangle([(self.half_trace_margin, self.half_trace_margin), ((self.width + self.half_trace_margin, self.height + self.half_trace_margin))], fill=None, outline=None, width=1)
        self.track_draw.line([(self.width + self.half_trace_margin - size_x, 0), (self.width + self.half_trace_margin - size_x, self.height + self.trace_margin)], fill="red", width=1, joint=None)
        self.track_draw.line([(0, self.height + self.half_trace_margin - size_y), (self.width + self.trace_margin, self.height + self.half_trace_margin - size_y)], fill="red", width=1, joint=None)
        self.trace_img_name = f"{tracking_path}tracing-{word_to_write}-{time.time()}.png"


# --- Static Integral Image for mask-based availability checks ---
class StaticIntegralImage:
    """
    Integral image built from a static availability mask.

    The mask uses 1 for available pixels and 0 for blocked pixels. The integral
    image stores cumulative counts of blocked pixels to allow O(1) queries that
    a rectangular region is free. Updates are supported with an incremental
    algorithm that only recomputes the affected region.
    """

    def __init__(self, height: int, width: int, mask_array: np.ndarray | None = None):
        self.height = height
        self.width = width

        # Occupancy grid: 1 = blocked/unavailable, 0 = free
        if mask_array is None:
            self._occupancy = np.zeros((height, width), dtype=np.uint8)
        else:
            if mask_array.shape != (height, width):
                raise ValueError(
                    f"Mask array shape {mask_array.shape} does not match provided dimensions ({height}x{width})"
                )
            # Mask uses 1 for available; invert to blocked
            self._occupancy = (mask_array == 0).astype(np.uint8)

        self.integral = self._compute_integral(self._occupancy)
        self._pending_updates = 0  # Track updates for batch recompute optimization

    @staticmethod
    def _compute_integral(grid: np.ndarray) -> np.ndarray:
        """Return summed-area table (height+1, width+1) for the blocked grid."""
        integral = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), dtype=np.uint64)
        integral[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
        return integral

    def is_valid_position(self, pos_y: int, pos_x: int, size_y: int, size_x: int) -> bool:
        """
        True when the rectangle fits in bounds and does not overlap blocked pixels.
        """
        if (
            pos_x < 0
            or pos_y < 0
            or size_x <= 0
            or size_y <= 0
            or pos_x + size_x > self.width
            or pos_y + size_y > self.height
        ):
            return False

        blocked = (
            self.integral[pos_y + size_y, pos_x + size_x]
            + self.integral[pos_y, pos_x]
            - self.integral[pos_y + size_y, pos_x]
            - self.integral[pos_y, pos_x + size_x]
        )
        return blocked == 0

    def update(self, occupied_mask: np.ndarray, x: int, y: int) -> None:
        """
        Mark additional blocked area using an occupancy mask.
        
        Uses incremental update for efficiency - only recomputes the integral
        image from the affected region downward and rightward.
        
        Args:
            occupied_mask: 2D array where non-zero values mark blocked pixels
            x: X-coordinate of top-left corner of the mask region
            y: Y-coordinate of top-left corner of the mask region
        """
        h, w = occupied_mask.shape
        end_y = min(self.height, y + h)
        end_x = min(self.width, x + w)
        if end_y <= y or end_x <= x:
            return
        view_h = end_y - y
        view_w = end_x - x
        
        # Update occupancy grid
        self._occupancy[y:end_y, x:end_x] = np.maximum(
            self._occupancy[y:end_y, x:end_x], occupied_mask[:view_h, :view_w].astype(np.uint8)
        )
        
        # Incremental update: recompute only the affected portion
        # This is O((H-y) * (W-x)) instead of O(H*W)
        self._incremental_update(y, x)
    
    def _incremental_update(self, start_y: int, start_x: int) -> None:
        """
        Incrementally update the integral image from (start_y, start_x).
        
        Uses the standard integral image formula:
        I(y,x) = img(y,x) + I(y-1,x) + I(y,x-1) - I(y-1,x-1)
        
        Only recomputes cells at and after (start_y, start_x).
        """
        # Recompute affected rows and columns
        for row in range(start_y, self.height):
            for col in range(start_x, self.width):
                # Integral index is offset by 1
                iy, ix = row + 1, col + 1
                self.integral[iy, ix] = (
                    self._occupancy[row, col]
                    + self.integral[iy - 1, ix]
                    + self.integral[iy, ix - 1]
                    - self.integral[iy - 1, ix - 1]
                )
        
        # Also need to update remaining columns in affected rows before start_x
        # when start_x > 0 and rows below start_y are affected
        if start_y > 0:
            for row in range(start_y, self.height):
                for col in range(start_x):
                    iy, ix = row + 1, col + 1
                    self.integral[iy, ix] = (
                        self._occupancy[row, col]
                        + self.integral[iy - 1, ix]
                        + self.integral[iy, ix - 1]
                        - self.integral[iy - 1, ix - 1]
                    )
    
    def full_recompute(self) -> None:
        """Force a full recomputation of the integral image."""
        self.integral = self._compute_integral(self._occupancy)
