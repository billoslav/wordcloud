import numpy as np
from random import randint
from operator import itemgetter
from PIL import Image, ImageFont, ImageDraw
from scipy import spatial
import quads
import matplotlib.pyplot as plt
import math
import time
import os

STRATEGIES = ["random", "brute", "archimedian", "rectangular", "archimedian_reverse", "rectangular_reverse", "KDTree", "quad", "pytag", "pytag_reverse"]

class IntegralImage(object):
    def __init__(self, height, width, tracing=False):
        self.height = height
        self.width = width
        
        self.tracing = tracing
        self.structure_created = False
        self.directory_name = "Tracking"
        
        # We are not using any mask, so the initial Image is filled with zeros
        self.integral = np.zeros((height, width), dtype=np.uint32)

    def find_position(self, size_x, size_y, place_strategy="random", word_to_write = ""):
        height = self.height
        width = self.width
        
        height_y = (height - size_y) // 2
        width_x = (width - size_x) // 2
        
        free_locations = []
        
        trace_img = self.tracing
        trace_margin = 200
        half_trace_margin = trace_margin/2
        
        INCREASE = 5
        DEFAULT_STEP = 2
        
        if trace_img:
            if self.structure_created:
                tracking_path = f"{self.directory_name}/{place_strategy}/"
            else:
                tracking_path = self.create_tracking_structure(self.directory_name, place_strategy)
                self.structure_created = True

            tracking_img = Image.new("L", (width+trace_margin, height+trace_margin), color="white")
            track_draw = ImageDraw.Draw(tracking_img)
            
            track_draw.rectangle([(half_trace_margin, half_trace_margin), ((width+half_trace_margin, height+half_trace_margin))], fill=None, outline=None, width=1)
            track_draw.line([(width+half_trace_margin-size_x, 0), (width+half_trace_margin-size_x, height+trace_margin)], fill="red", width=1, joint=None)
            track_draw.line([(0, height+half_trace_margin-size_y), (width+trace_margin, height+half_trace_margin-size_y)], fill="red", width=1, joint=None)
            trace_img_name = f"{tracking_path}tracing-{word_to_write}-{time.time()}.png"
    
        def is_valid_position(pos_y, pos_x):
            area = self.integral[pos_y, pos_x] + self.integral[pos_y + size_y, pos_x + size_x]
            area -= self.integral[pos_y + size_y, pos_x] + self.integral[pos_y, pos_x + size_x]
            
            return not area
        
        def check_bounds(x, y):
            conditions = [
                x > (width - size_x),
                y > (height - size_y),
                y < 0,
                x < 0
            ]
            
            return sum(conditions) >= 2
        
        def draw_trace_point(pos_x, pos_y):
            if trace_img: 
                track_draw.point([(pos_x + half_trace_margin, pos_y + half_trace_margin)], fill="red")
            
        def save_trace_img():
            if trace_img:
                tracking_img.save(trace_img_name)
        
        def random(width_x, height_y):
            return free_locations[randint(0, len(free_locations) - 1)]
        
        def rectangular(width_x, height_y):
            direction = 0
    
            for n in range(max(width, height)):
                draw_trace_point(width_x, height_y)
                
                if (width_x, height_y) in free_locations:
                    save_trace_img()
                    return width_x, height_y
        
                if check_bounds(width_x, height_y):
                    break
                    
                direction = n % 4
                axis = n % 2
                
                where_to_next = [
                    (width_x + INCREASE + n, height_y, DEFAULT_STEP), # right
                    (width_x, height_y + INCREASE + n, DEFAULT_STEP), # up
                    (width_x - INCREASE - n, height_y, -DEFAULT_STEP), # left
                    (width_x, height_y - INCREASE - n, -DEFAULT_STEP) # down
                ]

                end_x, end_y, defined_step = where_to_next[direction]
                    
                start_point, stop_point = (width_x, end_x) if (width_x != end_x) else (height_y, end_y)
                
                for current_position in range(start_point, stop_point, defined_step):
                    position_x, position_y = (current_position, height_y) if (axis == 0) else (width_x, current_position)
                    
                    draw_trace_point(position_x, position_y)
                        
                    if (position_x, position_y) in free_locations:
                        save_trace_img()
                        return position_x, position_y
                    
                width_x, height_y = end_x, end_y
                
            save_trace_img()
            return None
            # return rectangular_code(width_x, height_y, reverse=False)
        
        def rectangular_reverse(width_x, height_y):
            height_y = 0
            width_x = 0
            max_width = width - size_x
            max_height = height - size_y
            
            direction = 0
    
            for n in range(max(width, height)):
                draw_trace_point(width_x, height_y)
                
                if (width_x, height_y) in free_locations:
                    save_trace_img()
                    return width_x, height_y
        
                if check_bounds(width_x, height_y):
                    break
                    
                direction = n % 4
                axis = n % 2
                
                where_to_next = [
                    (max_width - width_x - INCREASE, height_y, DEFAULT_STEP), # right
                    (width_x, max_height - height_y - INCREASE, DEFAULT_STEP), # down
                    (max_width - width_x + INCREASE, height_y, -DEFAULT_STEP), # left
                    (width_x, max_height - height_y + INCREASE, -DEFAULT_STEP) # up
                ]

                end_x, end_y, defined_step = where_to_next[direction]
                    
                start_point, stop_point = (width_x, end_x) if (width_x != end_x) else (height_y, end_y)
                
                for current_position in range(start_point, stop_point, defined_step):
                    position_x, position_y = (current_position, height_y) if (axis == 0) else (width_x, current_position)
                    
                    draw_trace_point(position_x, position_y)
                        
                    if (position_x, position_y) in free_locations:
                        save_trace_img()
                        return position_x, position_y
                    
                width_x, height_y = end_x, end_y
                
            save_trace_img()
            return None
        
            # return rectangular_code(width_x, height_y, reverse=True)
        
        def rectangular_code(width_x, height_y, reverse=False):
            direction = 0
            width_code, height_code = width_x, height_y
            max_width = width - size_x
            max_height = height - size_y
    
            for n in range(max(width, height)):
                draw_trace_point(width_code, height_code)
                
                if (width_code, height_code) in free_locations:
                    save_trace_img()
                    return width_code, height_code
        
                if check_bounds(width_code, height_code):
                    break
                    
                direction = n % 4
                axis = n % 2
                
                where_to = [
                    (width_x + INCREASE + n, height_y, DEFAULT_STEP), # right
                    (width_x, height_y + INCREASE + n, DEFAULT_STEP), # up
                    (width_x - INCREASE - n, height_y, -DEFAULT_STEP), # left
                    (width_x, height_y - INCREASE - n, -DEFAULT_STEP) # down
                ]
                
                where_to_reverse = [
                    (max_width - width_x - INCREASE, height_y, DEFAULT_STEP), # right
                    (width_x, max_height - height_y - INCREASE, DEFAULT_STEP), # down
                    (max_width - width_x + INCREASE, height_y, -DEFAULT_STEP), # left
                    (width_x, max_height - height_y + INCREASE, -DEFAULT_STEP) # up
                ]
                
                where_to_next = where_to if not reverse else where_to_reverse
                
                end_x, end_y, defined_step = where_to_next[direction]
                    
                start_point, stop_point = (width_code, end_x) if (width_code != end_x) else (height_code, end_y)
                
                for current_position in range(start_point, stop_point, defined_step):
                    position_x, position_y = (current_position, height_code) if (axis == 0) else (width_code, current_position)
                    
                    draw_trace_point(position_x, position_y)
                        
                    if (position_x, position_y) in free_locations:
                        save_trace_img()
                        return position_x, position_y
                    
                width_code, height_code = end_x, end_y
                
            save_trace_img()
            return None
        
        def archimedian(width_x, height_y):
            e = width/height
             
            for n in range(height * width):
                draw_trace_point(width_x, height_y)
                    
                if check_bounds(width_x, height_y):
                    break
                    
                if (width_x, height_y) in free_locations:
                    save_trace_img()
                    return width_x, height_y
        
                width_x = width_x + int(e * (n* .1) * np.cos(n))
                height_y = height_y + int((n* .1) * np.sin(n))
                
            save_trace_img()
            return None
        
        def archimedian_reverse(width_x, height_y):
            spacing=0.5 # Distance between turns of the spiral.
            density=0.05 # Density of points along the spiral.
            
            max_radius = math.sqrt(width_x**2 + height_y**2)

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
                
                draw_trace_point(x, y)
        
                if (x, y) in free_locations:
                    save_trace_img()
                    return x, y

                # Decrease theta to move inward along the spiral
                theta -= density

            save_trace_img()
            return None
        
        def KDTree(width_x, height_y):
        
            (x, y) = free_locations[spatial.KDTree(free_locations).query([width_x, height_y])[1]]
        
            if (x, y):
                return x, y
            else: 
                return None
            
        def quad(width_x, height_y):
            tree = quads.QuadTree((width_x, height_y), width, height)
        
            for n in free_locations:
                tree.insert(n)
            
            point = tree.nearest_neighbors((width_x, height_y), count=1)
                
            if not point:
                return None
        
            return point[0].x, point[0].y
        
        def pytag(width_x, height_y):
            return pytag_code(width_x, height_y, False)
        
        def pytag_reverse(width_x, height_y):
            return pytag_code(width_x, height_y, True)
        
        def pytag_code(width_x, height_y, is_reverse=False):
            #https://github.com/atizo/PyTagCloud
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            
            if is_reverse: 
                directions.reverse()
                
            direction = directions[0]

            spl = 1
            
            while spl <= max(height, width):
                for step in range(spl * 2):
                    if step == spl:
                        direction = directions[(spl - 1) % 4]
                            
                    width_x = width_x + direction[0] * DEFAULT_STEP
                    height_y = height_y + direction[1] * DEFAULT_STEP
                    draw_trace_point(width_x, height_y)
                        
                    if (width_x, height_y) in free_locations:
                        save_trace_img()
                        return width_x, height_y
                        
                spl += 1
                    
            save_trace_img()
            return None
                    
        for line_y in range(height - size_y):
            for line_x in range(width - size_x):
            
                if is_valid_position(line_y, line_x):
                    
                    # If we check Brute force here, it gets aprox. 40% faster
                    if place_strategy == "brute":
                        return (line_x, line_y)
                    
                    free_locations.append((line_x, line_y))

        # If we cannot find any location, return None
        if not free_locations:
            return None
        
        if place_strategy not in STRATEGIES:
            raise ValueError(f"Incorrect placing strategy! The '{place_strategy}' is not defined.")
        else:
            return locals()[place_strategy](width_x, height_y)
                
    def update(self, new_img, x, y):
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
        structure = f"{parent_name}/{folder_name}" if parent_name else folder_name
        if not os.path.isdir(structure):
            try:
                os.mkdir(structure)
                print(f"Directory '{structure}' created successfully.")
            except FileExistsError:
                print(f"Directory '{structure}' already exists.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{structure}'.")
            except Exception as e:
                print(f"An error occurred: {e}")
    
    def create_tracking_structure(self, directory, place_strategy):
        self.create_folder(directory)
        self.create_folder(place_strategy, directory)
        
        return f"{directory}/{place_strategy}/"

class Wordcloud(object):
    def __init__(self, width=600, height=338, font_path=None, margin=2,
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
    
    def split_text(self, text_to_split, stopwords=None, min_word_length=None):
        splitted = text_to_split.split()
        res = {}
        
        self.stopwords = stopwords if stopwords is not None else self.stopwords
        self.min_word_length = min_word_length if min_word_length is not None else self.min_word_length
            
        for word in splitted:
            word = ''.join([i for i in word if i.isalpha()])
        
            if len(word) < self.min_word_length: word = ''
            if word in self.stopwords: word = ''
        
            if word: res[str(word).casefold()] = 1 if res.get(str(word).casefold()) == None else res[str(word).casefold()] + 1
    
        return res 
    
    def sort_normalize(self, input_words):
        frequencies = sorted(input_words.items(), key=itemgetter(1), reverse=True)
        
        if len(frequencies) <= 0:
            raise ValueError(f"We need at least 1 word, but got: '{len(frequencies)}'")
    
        max_frequency = float(frequencies[0][1])
        frequencies = [(word, freq / max_frequency, freq) for word, freq in frequencies]
    
        return frequencies
    
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
            img.save(f"{image_name}.png", optimize=True)
            
        return img
    
    def draw_plt_image(self):
        
        img = self.draw_image()
        
        plt.figure()
        plt.imshow(img, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        return self
        
    def generate_svg(self, save_file=False, file_name="svg_img"):
        
        height = self.height
        width = self.width
        max_font_size = self.max_font_size

        result = []

        # Get font information
        font = ImageFont.truetype(self.font_path, max_font_size)
        raw_font_family, raw_font_style = font.getname()
        
        font_family = repr(raw_font_family)
        raw_font_style = raw_font_style.lower()
        
        # ready for improvemets
        font_style = 'normal'
        font_weight = 'normal'

        # Header
        result.append(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">")

        # Style
        result.append(f"<style>text{{font-family:{font_family};font-weight:{font_weight};font-style:{font_style};}}</style>")

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
        
        if save_file:
            to_save = open(f"{file_name}.html", "a")
            to_save.write('\n'.join(result))
            to_save.close()
            
        return '\n'.join(result)
    
    def create_html(self, svg_content, save_file=False, file_name="wordcloud"):
        
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
            to_save = open(f"{file_name}.html", "a")
            to_save.write('\n'.join(html_file))
            to_save.close()
        
        return '\n'.join(html_file)
        