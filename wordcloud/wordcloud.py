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
    Represents an integral image for efficient area calculations.
    """
    def __init__(self, height, width, tracing=False):
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
        Finds a suitable position for a word using the specified strategy.
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
        """Checks if a position is free using the integral image."""
        if pos_y < 0 or pos_x < 0 or size_y < 0 or size_x < 0:
            raise ValueError("Negative size or coordinates of the image!")
        
        area = self.integral[pos_y, pos_x] + self.integral[pos_y + size_y, pos_x + size_x]
        area -= self.integral[pos_y + size_y, pos_x] + self.integral[pos_y, pos_x + size_x]
            
        return not area
        
    def check_bounds(self, x, y, size_x, size_y):
        """Checks if a position is out of bounds."""
        return sum([x > (self.width - size_x), y > (self.height - size_y), y < 0, x < 0]) >= 2
        
    def draw_trace_point(self, pos_x, pos_y):
        """Draws a point on the tracking image."""
        if self.tracing:
            self.track_draw.point([(pos_x + self.half_trace_margin, pos_y + self.half_trace_margin)], fill="red")

    def save_trace_img(self):
        """Saves the tracking image."""
        if self.tracing:
            self.tracking_img.save(self.trace_img_name)
    
    # Define placement strategies as inner functions
    def random(self, free_locations, width_x, height_y, size_x, size_y):
        return free_locations[randint(0, len(free_locations) - 1)]

    def rectangular_code(self, free_locations, width_x, height_y, size_x, size_y, reverse=False):  # Combined rectangular logic
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
    rectangular_reverse = lambda self, free_locations, width_x, height_y, size_x, size_y: self.rectangular_code(free_locations, 0, 0, size_x, size_y, reverse=True)

    def archimedian(self, free_locations, width_x, height_y, size_x, size_y):
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
        """Finds a nearby free location using a KDTree."""
        if not free_locations: return None
        
        (x, y) = free_locations[spatial.KDTree(free_locations).query([width_x, height_y])[1]]

        if (x, y):
            return x, y
        else: 
            return None
            
    def quad(self, free_locations, width_x, height_y, size_x, size_y):
        """Finds a nearby free location using a QuadTree."""
        if not free_locations: return None
        
        tree = quads.QuadTree((width_x, height_y), self.width, self.height)
            
        for n in free_locations:
            tree.insert(n)

        point = tree.nearest_neighbors((width_x, height_y), count=1)
                
        if not point:
            return None
        
        return point[0].x, point[0].y
        
    def pytag_code(self, free_locations, width_x, height_y, size_x, size_y, is_reverse=False):
        """Implements the PyTag placement strategy."""
        #https://github.com/atizo/PyTagCloud
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
    pytag_reverse = lambda self, free_locations, width_x, height_y, size_x, size_y: self.pytag_code(free_locations, width_x, height_y, size_x, size_y, is_reverse=True)
                
    #TODO - check Image size matching the origin size
    def update(self, new_img, x, y):
        """Updates the integral image efficiently."""
        if (x or y) < 0:
            raise ValueError("Negative size of the image!")
        
        if x > self.width or y > self.height:
            return
        
        # Use vectorized operations for faster calculation
        recomputed = np.cumsum(np.cumsum(new_img[y:, x:],axis=1), axis=0)

        # Apply existing integral image values using vectorized operations
        if y > 0:
            if x > 0:
                recomputed += (self.integral[y - 1, x:] - self.integral[y - 1, x - 1])
            else:
                recomputed += self.integral[y - 1, x:]
        if x > 0:
            recomputed += self.integral[y:, x - 1][:, np.newaxis]

        self.integral[y:, x:] = recomputed

    def create_folder(self, folder_name, parent_name=""):
        """Creates a directory if it doesn't exist, handling potential errors."""
        path = os.path.join(parent_name, folder_name) if parent_name else folder_name  # Use os.path.join
        try:
            os.makedirs(path, exist_ok=True)  # Use makedirs and exist_ok for cleaner creation
            print(f"Directory '{path}' created/already exists.")  # Informative message
        except OSError as e:  # Catch potential OS errors
            print(f"Error creating directory '{path}': {e}")

    def create_tracking_structure(self, directory, place_strategy):
        """Creates the tracking directory structure."""
        self.create_folder(directory)
        self.create_folder(place_strategy, directory)
        
        return f"{directory}/{place_strategy}/"
    
    def tracing_setup(self, size_x, size_y, place_strategy, word_to_write):
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
    def __init__(self, width=600, height=338, font_path="fonts/Arial Unicode.ttf", margin=2,
                 max_words=200, min_word_length=3,
                 min_font_size=14, max_font_size=None, font_step=2,
                 stopwords=[],  
                 background_color='white', mode="RGB", black_white = False, 
                 place_strategy='random', rect_only=False, tracing_files = False):
        
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
        """Splits and preprocesses text, counting word frequencies."""
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
        """Sorts word frequencies and normalizes them."""
        if not input_words:  # Check for empty input
            raise ValueError("No words to process.")

        frequencies = sorted(input_words.items(), key=itemgetter(1), reverse=True)
        max_frequency = float(frequencies[0][1])
        return [(word, freq / max_frequency, freq) for word, freq in frequencies]

    def prepare_text(self, to_split, stopwords=None, min_word_length=None):
        splitted = self.split_text(to_split, stopwords, min_word_length)
        
        return self.sort_normalize(splitted)

    def find_position(self, frequencies):
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
        self.font_size = None
    
        integral_image = IntegralImage(self.height, self.width)

        # create control image
        control_img = Image.new("L", (self.width, self.height))
        draw = ImageDraw.Draw(control_img)
        
        #prepare new variables we want to resave for each word
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
        for ids, ((word, freq, count), font_path, word_font_size, position, orientation, color) in enumerate(self.gen_positions):
            if word in new_colors.keys():
                self.gen_positions[ids] = (word, freq, count), font_path, word_font_size, position, orientation, new_colors[word]
            
        return self
    
    def generate(self, text_to_analyze):
        normalized_and_sorted = self.prepare_text(text_to_analyze)
        
        return self.find_position(normalized_and_sorted)
    
    def draw_image(self, save_file=False, image_name="wordcloud"):
        
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
        
        img = self.draw_image()
        
        plt.figure()
        plt.imshow(img, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        return self

    def generate_svg(self, save_file=False, file_name="svg_img"):
        """Generates SVG content for the wordcloud."""

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
        """Creates an HTML file with the SVG content and tooltip functionality."""
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
        """Creates a directory if it doesn't exist, handling potential errors."""
        try:
            os.makedirs(folder_name, exist_ok=True)  # Use makedirs and exist_ok for cleaner creation
            print(f"Directory '{folder_name}' created/already exists.")  # Informative message
        except OSError as e:  # Catch potential OS errors
            print(f"Error creating directory '{folder_name}': {e}")
