import numpy as np
from random import randint
from operator import itemgetter
from PIL import Image, ImageFont, ImageDraw
from scipy import spatial
import quads
import matplotlib.pyplot as plt

STRATEGIES = ["random", "brute", "archimedian", "rectangular", "archimedian_reverse", "rectangular_reverse", "KDTree", "quad", "pytag", "pytag_reverse"]

class IntegralImage(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        
        # We are not using any mask, so the initial Image is filled with zeros
        self.integral = np.zeros((height, width), dtype=np.uint32)

    def find_position(self, size_x, size_y, place_strategy="random"):
        height = self.height
        width = self.width
        free_locations = []
        
        INCREASE = 5
        DEFAULT_STEP = 2
        
        if place_strategy not in STRATEGIES:
            raise ValueError(f"Incorrect placing strategy! The '{place_strategy}' is not defined.")
    
        def is_valid_position(i, j):
            area = self.integral[i, j] + self.integral[i + size_x, j + size_y]
            area -= self.integral[i + size_x, j] + self.integral[i, j + size_y]
            return not area
        
        def check_bounds(x, y):
            return (y >= self.width or x >= self.height or y < 0 or x < 0)

        for i in range(height - size_x):
            for j in range(width - size_y):
                if is_valid_position(i, j):
                    
                    # If we check Brute force here, it gets aprox. 40% faster
                    if place_strategy == "brute":
                        return (i, j)
                    
                    free_locations.append((i, j))

        # If we cannot find any location, return None
        if not free_locations:
            return None
        
        match place_strategy:
            case "random":
                return free_locations[randint(0, len(free_locations) - 1)]
            
            case "rectangular":
                height_x = (height - size_x) // 2
                width_y = (width - size_y) // 2
    
                direction = 0
    
                for n in range(max(height, width)):

                    if (height_x, width_y) in free_locations:
                        return height_x, width_y
        
                    if check_bounds(height_x, width_y):
                        continue
        
                    direction = n % 4
                    
                    if direction == 0:
                        for y in range(width_y, width_y + INCREASE + n, DEFAULT_STEP):
                            if check_bounds(height_x, y):
                                break
                
                            if (height_x, y) in free_locations:
                                return height_x, y
                
                        width_y += INCREASE + n

                    elif direction == 1:
                        for x in range(height_x, height_x + INCREASE + n, DEFAULT_STEP):
                            if check_bounds(x, width_y):
                                break
                            
                            if (x, width_y) in free_locations:
                                return x, width_y
                
                        height_x += INCREASE + n

                    elif direction == 2:
                        for y in range(width_y,  width_y - INCREASE - n, -DEFAULT_STEP):
                            if check_bounds(height_x, y):
                                break
                            
                            if (height_x, y) in free_locations:
                                return height_x, y
                
                        width_y -= INCREASE + n

                    else:
                        for x in range(height_x, height_x - INCREASE - n, -DEFAULT_STEP):
                            if check_bounds(x, width_y):
                                break
                            
                            if (x, width_y) in free_locations:
                                return x, width_y
                
                        height_x -= INCREASE + n
                        
                return None
            
            case "rectangular_reverse":
                height_x = 0
                width_y = 0
    
                direction = 0
    
                for n in range(max(height, width)):
        
                    if (height_x, width_y) in free_locations:
                        return height_x, width_y
        
                    direction = n % 4
            
                    if check_bounds(height_x, width_y):
                        continue

                    if direction == 0:
                        for y in range(width_y, width - width_y, DEFAULT_STEP):
                            if check_bounds(height_x, y):
                                break
                            
                            if (height_x, y) in free_locations:
                                return height_x, y
                
                        width_y = width - width_y - INCREASE

                    elif direction == 1:
                        for x in range(height_x, height - height_x, DEFAULT_STEP):
                            if check_bounds(x, width_y):
                                break
                            
                            if (x, width_y) in free_locations:
                                return x, width_y
                
                        height_x = height - height_x - INCREASE

                    elif direction == 2:
                        for y in range(width_y, width - width_y, -DEFAULT_STEP):
                            if check_bounds(height_x, y):
                                break
                            
                            if (height_x, y) in free_locations:
                                return height_x, y
                
                        width_y = width - width_y + INCREASE

                    else:
                        for x in range(height_x, height - height_x, -DEFAULT_STEP):
                            if check_bounds(x, width_y):
                                break
                            
                            if (x, width_y) in free_locations:
                                return x, width_y
                
                        height_x = height - height_x + INCREASE
                        
                return None
                        
            case "archimedian":
                e = width/height
                height_x = (height - size_x) // 2
                width_y = (width - size_y) // 2
    
                for n in range(height * width):
                    if (height_x, width_y) in free_locations:
                        return height_x, width_y
        
                    width_y += int(e * (n* .1) * np.cos(n))
                    height_x += int((n* .1) * np.sin(n))
                
                return None
            
            case "archimedian_reverse":
                e = width/height
                height_x = (height - size_x) // 2
                width_y = (width - size_y) // 2
                
                for n in range((height_x * width_y)//2, 0, -1):
                    width_y -= int(e * (n* .1) * np.cos(n))
                    height_x -= int((n* .1) * np.sin(n))
                    
                    if check_bounds(height_x, width_y):
                        continue
                    
                    if (height_x, width_y) in free_locations:
                        return height_x, width_y
                
                return None
            
            case "KDTree":
                height_x = height // 2
                width_y = width // 2
        
                (x, y) = free_locations[spatial.KDTree(free_locations).query([height_x, width_y])[1]]
        
                if (x, y):
                    return x, y
                else: 
                    return None
                
            case "quad":
                height_x = height // 2
                width_y = width // 2
        
                tree = quads.QuadTree((height_x ,width_y), height, width)
        
                for n in free_locations:
                    tree.insert(n)
            
                point = tree.nearest_neighbors((height_x, width_y), count=1)
                
                if not point:
                    return None
        
                return point[0].x, point[0].y
            
            case "pytag":
                #https://github.com/atizo/PyTagCloud
                STEP_SIZE = 2
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                
                direction = directions[0]

                spl = 1
                width_y = (width - size_y) // 2
                height_x = (height - size_x) // 2
            
                while spl <= max(height, width):
                    for step in range(spl * 2):
                        if step == spl:
                            direction = directions[(spl - 1) % 4]
                            
                        width_y += direction[0] * STEP_SIZE * DEFAULT_STEP
                        height_x += direction[1] * STEP_SIZE * DEFAULT_STEP
                        
                        if (height_x, width_y) in free_locations:
                            return height_x, width_y
                        
                    spl += 1
                    
                return None
            
            case "pytag_reverse":
                #https://github.com/atizo/PyTagCloud
                STEP_SIZE = 2
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                directions.reverse()
                
                direction = directions[0]

                spl = 1
                width_y = (width - size_y) // 2
                height_x = (height - size_x) // 2
            
                while spl <= max(height, width):
                    for step in range(spl * 2):
                        if step == spl:
                            direction = directions[(spl - 1) % 4]
                            
                        width_y += direction[0] * STEP_SIZE * DEFAULT_STEP
                        height_x += direction[1] * STEP_SIZE * DEFAULT_STEP
                        
                        if (height_x, width_y) in free_locations:
                            return height_x, width_y
                        
                    spl += 1
                    
                return None
            
            case _:
                # Already checked at the beggining of the function, but just in case
                raise ValueError(f"Incorrect placing strategy defined! The {place_strategy} cannot be used.")
                
    def update(self, new_img, x, y):
        recomputed = np.cumsum(np.cumsum(new_img[x:, y:],axis=1), axis=0)
        
        if x > 0:
            if y > 0:
                recomputed += (self.integral[x - 1, y:] - self.integral[x - 1, y - 1])
            else:
                recomputed += self.integral[x - 1, y:]
        if y > 0:
            recomputed += self.integral[x:, y - 1][:, np.newaxis]

        self.integral[x:, y:] = recomputed

class Wordcloud(object):
    def __init__(self, width=600, height=338, font_path=None, margin=2,
                 max_words=200, min_word_length=3,
                 min_font_size=14, max_font_size=None, font_step=2,
                 stopwords=[],  
                 background_color='white', mode="RGB", black_white = False, 
                 place_strategy='random', rect_only=False):
        
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
    
        integral_image = IntegralImage(self.height, self.width)

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
                if len(frequencies) == 1:
                    self.max_font_size = self.height
                else:
                    self.max_font_size = self.def_max_font_size
            
            # select the font size
            if self.font_size:
                self.font_size = min(self.font_size, int(round(freq * self.max_font_size)))
            else: 
                self.font_size = int(round(freq * self.max_font_size))
        
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
                result = integral_image.find_position(box_size[3] + self.margin, box_size[2] + self.margin, self.place_strategy)
                
                # found a place
                if result is not None:
                    break
                
                # we didn't find a place, make font smaller and try again
                self.font_size -= self.font_step
            
            # font_size is too small
            if self.font_size < self.min_font_size:
                break

            height_x, width_y = np.array(result) + self.margin // 2
            
            # draw the text to control img
            draw.text((width_y, height_x), word, fill="white", font=transposed_font)
            
            if self.rect_only:
                bbox = draw.textbbox((width_y, height_x), word, font=transposed_font)
                draw.rectangle(bbox, outline="white")
                
            font_paths.append(self.font_path)
            positions.append((height_x, width_y))
            orientations.append(def_orientation)
            font_sizes.append(self.font_size)
            
            if self.black_white:
                colors.append("rgb(0, 0, 0)")
            else:
                colors.append(f"rgb({randint(0,256)}, {randint(0,256)}, {randint(0,256)})")
        
            # create numpy array for control image
            img_array = np.asarray(control_img)
            
            #update integral image with new word
            integral_image.update(img_array, height_x, width_y)
        
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
            if self.font_size:
                self.font_size = min(self.font_size, int(round(freq * self.max_font_size)))
            else: 
                self.font_size = int(round(freq * self.max_font_size))
        
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
                result = integral_image.find_position(box_size[3] + self.margin, box_size[2] + self.margin, self.place_strategy)
                
                # Found a place
                if result is not None:
                    break
                
                # we didn't find a place, make font smaller and try again
                self.font_size -= self.font_step
            
            # check font size
            if self.font_size < self.min_font_size:
                break

            height_x, width_y = np.array(result) + self.margin // 2
            
            # draw the text to control img
            draw.text((width_y, height_x), word, fill="white", font=transposed_font)
            
            if self.rect_only:
                bbox = draw.textbbox((width_y, height_x), word, font=transposed_font)
                draw.rectangle(bbox, outline="white")
                
            if word in new_fonts.keys():
                font_paths.append(new_fonts[word])
            else:
                font_paths.append(self.font_path)
            
            font_sizes.append(self.font_size)
            positions.append((height_x, width_y))
            new_freq.append((word, freq, count))
            
            # we are not changing orientation or colors
            orientations.append(orientation)
            colors.append(color)
            
            # create numpy array for control image
            img_array = np.asarray(control_img)
            
            #update integral image with new word
            integral_image.update(img_array, height_x, width_y)
            
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
            pos = (position[1], position[0])
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

        for (word, freq, count), font_path, font_size, (y, x), orientation, color in self.gen_positions:
           
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
        