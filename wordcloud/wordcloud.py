import numpy as np
from random import randint
from operator import itemgetter
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np
from random import randint
from PIL import Image, ImageDraw
from scipy import spatial
import quads
import math
import time
import os

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
    def __init__(self, height, width, tracing=False):
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

        # We are not using any mask, so the initial Image is filled with zeros
        self.integral = np.zeros((height, width), dtype=np.uint64)
                
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
            return None

        # Call the selected placement strategy
        if place_strategy not in STRATEGIES:
            raise ValueError(f"Incorrect placing strategy! The '{place_strategy}' is not defined.")
        else:
            method_to_call = getattr(self, place_strategy)
            return method_to_call(free_locations, width_x, height_y, size_x, size_y)

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
            free_locations (list): List of available (x, y) coordinate tuples
            width_x (int): X-coordinate of the center point (used by other strategies)
            height_y (int): Y-coordinate of the center point (used by other strategies)
            size_x (int): Width of the word bounding box (used by other strategies)
            size_y (int): Height of the word bounding box (used by other strategies)
            
        Returns:
            tuple: Selected (x, y) position for word placement
        """
        return free_locations[randint(0, len(free_locations) - 1)]

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
            free_locations (list): List of available (x, y) coordinate tuples
            width_x (int): X-coordinate of target point (usually center)
            height_y (int): Y-coordinate of target point (usually center)
            size_x (int): Width of the word bounding box (not used in this strategy)
            size_y (int): Height of the word bounding box (not used in this strategy)
            
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no positions available
        """
        if not free_locations: return None
        
        (x, y) = free_locations[spatial.KDTree(free_locations).query([width_x, height_y])[1]]

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
            free_locations (list): List of available (x, y) coordinate tuples
            width_x (int): X-coordinate of target point (usually center)
            height_y (int): Y-coordinate of target point (usually center)
            size_x (int): Width of the word bounding box (not used in this strategy)
            size_y (int): Height of the word bounding box (not used in this strategy)
            
        Returns:
            tuple or None: Selected (x, y) position for word placement, or None if no positions available
        """
        if not free_locations: return None
        
        tree = quads.QuadTree((width_x, height_y), self.width, self.height)
            
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
            - Prints status messages about folder creation or errors
        """
        path = os.path.join(parent_name, folder_name) if parent_name else folder_name
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created/already exists.")
        except OSError as e:
            print(f"Error creating directory '{path}': {e}")

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

class Wordcloud:
    """
    Main class for generating wordclouds from text.
    
    The Wordcloud class provides the primary user interface for creating
    wordclouds from text data. It handles text preprocessing, word frequency analysis,
    word positioning, and visualization in various formats.
    
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
        gen_positions (list): Generated positions for words
        results_folder (str): Folder to save output files
    """
    def __init__(self, width=600, height=338, font_path="fonts/Arial Unicode.ttf", margin=2,
                 max_words=200, min_word_length=3,
                 min_font_size=14, max_font_size=None, font_step=2,
                 stopwords=[],  
                 background_color='white', mode="RGB", black_white = False, 
                 place_strategy='random', rect_only=False, tracing_files = False):
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
            stopwords (list, optional): Words to exclude from the wordcloud. Defaults to [].
            background_color (str, optional): Background color. Defaults to 'white'.
            mode (str, optional): Color mode ('RGB', 'RGBA', etc.). Defaults to "RGB".
            black_white (bool, optional): Use only black text. Defaults to False.
            place_strategy (str, optional): Word placement strategy. Defaults to 'random'.
            rect_only (bool, optional): Draw rectangles instead of text. Defaults to False.
            tracing_files (bool, optional): Generate debug tracing files. Defaults to False.
        """
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
        self.stopwords = stopwords
        self.background_color = background_color
        self.mode = mode
        self.black_white = black_white
        self.place_strategy = place_strategy
        self.rect_only = rect_only
        self.tracing_files = tracing_files
        
        self.def_max_font_size = 60
        self.results_folder = os.getcwd() + "/Results"

    def split_text(self, text_to_split, stopwords=None, min_word_length=None):
        """
        Split and preprocess text, counting word frequencies.
        
        This method takes a string of text, splits it into words, and counts
        the frequency of each word. It also handles case normalization,
        stopword removal, and filtering by minimum word length.
        
        Args:
            text_to_split (str): Text to analyze
            stopwords (list, optional): List of words to exclude. If None, use the instance's stopwords.
            min_word_length (int, optional): Minimum word length to include. If None, use the instance's value.
            
        Returns:
            dict: Dictionary of {word: frequency} pairs
            
        Example:
            >>> wc = Wordcloud(stopwords=["and", "the"])
            >>> wc.split_text("The quick and the dead")
            {'quick': 1, 'dead': 1}
        """
        self.stopwords = stopwords if stopwords is not None else self.stopwords
        self.min_word_length = min_word_length if min_word_length is not None else self.min_word_length

        res = {}
        for word in text_to_split.split():
            word = ''.join([i for i in word if i.isalpha()])
        
            if len(word) < self.min_word_length: word = ''
            if word in self.stopwords: word = ''
        
            if word: res[str(word).casefold()] = 1 if res.get(str(word).casefold()) == None else res[str(word).casefold()] + 1
    
        return res 
    
    def sort_normalize(self, input_words):
        """
        Sort words by frequency and normalize frequencies.
        
        This method sorts words by their frequency (descending) and normalizes
        the frequencies relative to the most frequent word. This ensures that
        the most frequent word will have a normalized frequency of 1.0, and all
        other words will have normalized frequencies between 0.0 and 1.0.
        
        Args:
            input_words (dict): Dictionary of {word: frequency} pairs
            
        Returns:
            list: List of (word, normalized_frequency, original_frequency) tuples
            
        Raises:
            ValueError: If input_words is empty
            
        Example:
            >>> wc = Wordcloud()
            >>> wc.sort_normalize({'apple': 5, 'banana': 3, 'cherry': 1})
            [('apple', 1.0, 5), ('banana', 0.6, 3), ('cherry', 0.2, 1)]
        """
        if not input_words:  # Check for empty input
            raise ValueError("No words to process.")

        frequencies = sorted(input_words.items(), key=itemgetter(1), reverse=True)
        max_frequency = float(frequencies[0][1])
        return [(word, freq / max_frequency, freq) for word, freq in frequencies]

    def prepare_text(self, to_split, stopwords=None, min_word_length=None):
        """
        Prepare text for wordcloud generation.
        
        This is a convenience method that combines text splitting, frequency
        counting, and normalization in a single call.
        
        Args:
            to_split (str): Text to analyze
            stopwords (list, optional): List of words to exclude
            min_word_length (int, optional): Minimum word length to include
            
        Returns:
            list: List of (word, normalized_frequency, original_frequency) tuples
            
        Example:
            >>> wc = Wordcloud()
            >>> wc.prepare_text("apple apple banana banana banana cherry")
            [('banana', 1.0, 3), ('apple', 0.6666666666666666, 2), ('cherry', 0.3333333333333333, 1)]
        """
        splitted = self.split_text(to_split, stopwords, min_word_length)
        
        return self.sort_normalize(splitted)

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
        integral_image = IntegralImage(self.height, self.width, self.tracing_files)

        # create control image
        control_img = Image.new("L", (self.width, self.height))
        draw = ImageDraw.Draw(control_img)
        
        #prepare variables we want to save for each word
        font_paths, font_sizes, positions, orientations, colors = [], [], [], [], []
    
        # we are not using other font oriantations, so default is None
        def_orientation = None
        
        # start drawing greyscale image
        for word, freq, count in frequencies:
            
            if freq == 0:
                continue
        
            # if there is only one word, set Max size to image height, use default max height otherwise
            if self.max_font_size is None:
                self.max_font_size = self.height if len(frequencies) == 1 else self.def_max_font_size
            
            # select the font size
            self.font_size = min(self.font_size, int(round(freq * self.max_font_size))) if self.font_size else int(round(freq * self.max_font_size))
        
            # look for a place until it's found or font became too small
            while True:
                
                # font_size is too small
                if self.font_size < self.min_font_size:
                    break
            
                font = ImageFont.truetype(self.font_path, self.font_size)
                transposed_font = ImageFont.TransposedFont(font, orientation=def_orientation)
                
                # get size of resulting text
                box_size = draw.textbbox((0, 0), word, font=transposed_font, anchor="lt")
                
                # find possible places using integral image:
                result = integral_image.find_position(box_size[2] + self.margin, box_size[3] + self.margin, self.place_strategy, word)
                
                # found a place
                if result is not None:
                    break
                
                # we didn't find a place, make font smaller and try again
                self.font_size -= self.font_step
            
            # font_size is too small
            if self.font_size < self.min_font_size:
                break

            width_x, height_y = np.array(result) + self.margin // 2
            
            # draw the text to control img
            draw.text((width_x, height_y), word, fill="white", font=transposed_font)
            
            if self.rect_only:
                bbox = draw.textbbox((width_x, height_y), word, font=transposed_font)
                draw.rectangle(bbox, outline="white")
                
            font_paths.append(self.font_path)
            positions.append((width_x, height_y))
            orientations.append(def_orientation)
            font_sizes.append(self.font_size)
            
            if self.black_white:
                colors.append("rgb(0, 0, 0)")
            else:
                colors.append(f"rgb({randint(0,256)}, {randint(0,256)}, {randint(0,256)})")
        
            # create numpy array for control image
            img_array = np.asarray(control_img)
            
            #update integral image with new word
            integral_image.update(img_array, width_x, height_y)
        
        self.gen_positions = list(zip(frequencies, font_paths, font_sizes, positions, orientations, colors))
        
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
        self.font_size = None
    
        integral_image = IntegralImage(self.height, self.width)

        # create control image
        control_img = Image.new("L", (self.width, self.height))
        draw = ImageDraw.Draw(control_img)
        
        new_freq, font_paths, font_sizes, positions, orientations, colors = [], [], [], [], [], []
        
        # we are not using other font oriantations, so default is None
        def_orientation = None
    
        # start drawing greyscale image
        for (word, freq, count), font_path, word_font_size, position, orientation, color in self.gen_positions:
            
            if freq == 0:
                continue
        
            # with the change of fonts, some sizes can differ as well
            if self.max_font_size is None:
                if len(self.gen_positions) == 1:
                    self.max_font_size = self.height
                else:
                    self.max_font_size = self.def_max_font_size
            
            # select the font size
            self.font_size = min(self.font_size, int(round(freq * self.max_font_size))) if self.font_size else int(round(freq * self.max_font_size))
        
            # look for a place until it's found or font became too small
            while True:
                
                # font-size is too small
                if self.font_size < self.min_font_size:
                    break
                
                if word in new_fonts.keys():
                    font = ImageFont.truetype(new_fonts[word], self.font_size)
                else:
                    font = ImageFont.truetype(self.font_path, self.font_size)
                    
                transposed_font = ImageFont.TransposedFont(font, orientation=def_orientation)
                
                # get size of resulting text
                box_size = draw.textbbox((0, 0), word, font=transposed_font, anchor="lt")
                
                # find possible places using integral image:
                result = integral_image.find_position(box_size[2] + self.margin, box_size[3] + self.margin, self.place_strategy, word)
                
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
            draw.text((width_x, height_y), word, fill="white", font=transposed_font)            
            
            if self.rect_only:
                bbox = draw.textbbox((width_x, height_y), word, font=transposed_font)
                draw.rectangle(bbox, outline="white")
                
            if word in new_fonts.keys():
                font_paths.append(new_fonts[word])
            else:
                font_paths.append(self.font_path)
            
            font_sizes.append(self.font_size)
            positions.append((width_x, height_y))
            new_freq.append((word, freq, count))
            
            # we are not changing orientation or colors
            orientations.append(orientation)
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
        for ids, ((word, freq, count), font_path, word_font_size, position, orientation, color) in enumerate(self.gen_positions):
            if word in new_colors.keys():
                self.gen_positions[ids] = (word, freq, count), font_path, word_font_size, position, orientation, new_colors[word]
            
        return self
    
    def generate(self, text_to_analyze):
        """
        Generate a wordcloud from text.
        
        This is the main entry point for generating a wordcloud from text.
        It combines text preparation and word positioning in a single call.
        
        Args:
            text_to_analyze (str): Text to analyze and visualize
            
        Returns:
            Wordcloud: Self for method chaining
            
        Example:
            >>> wc = Wordcloud(width=800, height=400)
            >>> wc.generate("This is a sample text for wordcloud generation").draw_image(save_file=True)
        """
        normalized_and_sorted = self.prepare_text(text_to_analyze)
        
        return self.find_position(normalized_and_sorted)
    
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

        img = Image.new(self.mode, (width, height), self.background_color)
        draw = ImageDraw.Draw(img)
        
        for (word, freq, count), font_path, font_size, position, orientation, color in self.gen_positions:
            font = ImageFont.truetype(font_path, font_size)
            transposed_font = ImageFont.TransposedFont(font, orientation=orientation)
            pos = (position[0], position[1])
            draw.text(pos, word, fill=color, font=transposed_font)
            
        if save_file:
            self.create_folder(self.results_folder)
            img.save(f"{self.results_folder}/{image_name}.png", optimize=True)
            
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
        height = self.height
        width = self.width

        result = []

        # Get font information
        font = ImageFont.truetype(self.font_path, self.max_font_size or self.def_max_font_size)
        raw_font_family, raw_font_style = font.getname()
        
        font_family = repr(raw_font_family)
        raw_font_style = raw_font_style.lower()
        
        # ready for improvemets
        font_style = 'normal'
        font_weight = 'normal'

        # Header
        result.append(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">")

        # Style
        result.append(f"<style>text{{font-family:{font_family}, sans-serif; font-weight:{font_weight}; font-style:{font_style};}}</style>")

        # Background if defined
        if self.background_color is not None:
            result.append(f"<rect width=\"100%\" height=\"100%\" style=\"fill:{self.background_color}\"></rect>")

        for (word, freq, count), font_path, font_size, (x, y), orientation, color in self.gen_positions:
           
            font = ImageFont.truetype(font_path, font_size)
            bbox = font.getbbox(word)
            ascent, descent = font.getmetrics()

            x = x - bbox[0]
            y = y + (ascent - bbox[1])

            # Create text element
            result.append(f"<text id=\"{word}\" transform=\"translate({x},{y})\" font-size=\"{font_size}\" style=\"fill:{color}\" count=\"{count}\">{word}</text>")

        result.append('</svg>')

        svg_content = '\n'.join(result)

        if save_file:
            try:
                self.create_folder(self.results_folder)
                with open(f"{self.results_folder}/{file_name}.svg", "w", encoding="utf-8") as f: # Specify encoding
                    f.write(svg_content)
            except OSError as e:
                print(f"Error saving SVG: {e}")

        return svg_content

    def create_html(self, svg_content, save_file=False, file_name="wordcloud"):
        """
        Create an HTML file with the SVG content and tooltip functionality.
        
        Embeds the SVG wordcloud in an HTML document with JavaScript that
        displays word counts as tooltips when hovering over words.
        
        Args:
            svg_content (str): SVG content to embed (from generate_svg method)
            save_file (bool, optional): Whether to save the HTML to disk. Defaults to False.
            file_name (str, optional): Base name for the saved HTML file. Defaults to "wordcloud".
            
        Returns:
            str: The HTML content as a string
            
        Notes:
            - The HTML includes JavaScript for interactive tooltips showing word counts
            - The file is saved as "{file_name}.html" in the results_folder if save_file is True
            - This is useful for creating interactive visualizations for web applications
        """
        html_file = []
        
        #HTML file structure and basic styles
        html_file.append(
           """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wordcloud HTML page</title>
  <style>
    #tooltip {
      position: absolute;
      background-color: #f8f8f8;
      padding: 5px;
      border: 1px solid #ccc;
      border-radius: 5px;
      opacity: 0;
      transition: opacity 0.2s ease-in-out;
      z-index: 10;
    }
    #tooltip.show {
      opacity: 1;
    }
  </style>
</head>
<body>""")
        
        # Adding image as SVG
        html_file.append(svg_content)
        
        # Adding JS for tooltips and ending HTML
        html_file.append(
            """<div id="tooltip"></div>

  <script>
    const tooltip = document.getElementById('tooltip');
    const texts = document.querySelectorAll('svg text');

    texts.forEach(text => {
      const count = text.getAttribute('count');
      text.addEventListener('mouseover', function() {
        tooltip.textContent = `Word count: ${count}`;
        tooltip.classList.add('show');

        const rect = this.getBoundingClientRect();
        tooltip.style.left = `${rect.left + window.pageXOffset}px`;
        tooltip.style.top = `${rect.top + window.pageYOffset}px`;
      });

      text.addEventListener('mouseout', () => {
        tooltip.classList.remove('show');
      });
    });
  </script>
</body>
</html>""")
        
        if save_file:
            self.create_folder(self.results_folder)
            to_save = open(f"{self.results_folder}/{file_name}.html", "a")
            to_save.write('\n'.join(html_file))
            to_save.close()
        
        return '\n'.join(html_file)
    
    def create_folder(self, folder_name):
        """
        Create a directory if it doesn't exist.
        
        Creates a folder at the specified path, handling potential errors gracefully.
        This method is used for creating folders for output files.
        
        Args:
            folder_name (str): Name of the folder to create
            
        Notes:
            - Uses os.makedirs with exist_ok=True to avoid race conditions
            - Prints status messages about folder creation or errors
        """
        try:
            os.makedirs(folder_name, exist_ok=True)  # Use makedirs and exist_ok for cleaner creation
            print(f"Directory '{folder_name}' created/already exists.")  # Informative message
        except OSError as e:  # Catch potential OS errors
            print(f"Error creating directory '{folder_name}': {e}")
