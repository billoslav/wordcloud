# Wordcloud
## Master Thesis Project

A flexible and powerful Python library for creating visually appealing word clouds from text data.

## Features

- **Text Processing**: Extract and analyze word frequencies from any text
- **Customizable Appearance**: Control fonts, colors, sizes, and margins
- **Multiple Placement Strategies**: Various algorithms for positioning words including:
  - Random placement
  - Archimedean spirals (forward and reverse)
  - Rectangular spirals (forward and reverse)
  - KD-Tree optimization
  - Quad-Tree optimization
  - PyTagCloud inspired spirals
- **Multiple Export Formats**:
  - PNG images
  - SVG vector graphics
  - Interactive HTML with tooltips showing word counts
- **Advanced Customization**: Fine-tune font sizes, word counts, colors, and more

## Installation

```bash
pip install wordcloud
```

## Quick Start

```python
from wordcloud import Wordcloud

# Create a wordcloud instance
wc = Wordcloud(width=800, height=400, 
              font_path="fonts/Arial.ttf",
              background_color="white")

# Generate and display the wordcloud
wc.generate("Your text goes here...").draw_plt_image()

# Save the wordcloud as an image
wc.draw_image(save_file=True, image_name="my_wordcloud")
```

## Usage Examples

### Basic Word Cloud

```python
from wordcloud import Wordcloud

# Create a simple word cloud from a text string
text = "Python is an amazing programming language for data analysis, machine learning, web development, and more. Python has excellent libraries like NumPy, Pandas, TensorFlow, and of course this wordcloud generator!"

wc = Wordcloud()
wc.generate(text)
wc.draw_plt_image()  # Display the wordcloud
```

### Customized Word Cloud

```python
from wordcloud import Wordcloud

# Create a customized word cloud
wc = Wordcloud(
    width=1000,
    height=600,
    background_color="#333333",
    font_path="fonts/OpenSans-Bold.ttf",
    max_words=100,
    min_font_size=12,
    max_font_size=80,
    place_strategy="archimedian",
    stopwords=["and", "the", "to", "of", "a"]
)

# Read text from a file
with open("sample_text.txt", "r") as f:
    text = f.read()

wc.generate(text)
wc.draw_image(save_file=True, image_name="custom_wordcloud")
```

### Interactive HTML Word Cloud

```python
from wordcloud import Wordcloud

# Create a word cloud and export as interactive HTML
wc = Wordcloud(width=800, height=500)
wc.generate("Generate an interactive HTML wordcloud with tooltips showing word frequencies!")

# Generate SVG content
svg_content = wc.generate_svg()

# Create HTML with the SVG and interactive tooltips
html_content = wc.create_html(svg_content, save_file=True, file_name="interactive_wordcloud")
```

## Word Placement Strategies

The `place_strategy` parameter controls how words are positioned in the cloud:

- `"random"`: Random placement (fastest)
- `"brute"`: Checks all possible positions
- `"archimedian"`: Archimedean spiral from center outward
- `"archimedian_reverse"`: Archimedean spiral from outside inward
- `"rectangular"`: Rectangular spiral from center outward
- `"rectangular_reverse"`: Rectangular spiral from outside inward
- `"KDTree"`: Uses KD-Tree spatial data structure for efficient placement
- `"quad"`: Uses Quad-Tree spatial data structure for efficient placement
- `"pytag"`: PyTagCloud-inspired spiral
- `"pytag_reverse"`: Reverse PyTagCloud spiral

Example:
```python
# Create a wordcloud with an Archimedean spiral layout
wc = Wordcloud(place_strategy="archimedian")
```

## API Reference

### Class: Wordcloud

Main class for generating wordclouds from text.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| width | int | 600 | Width of the wordcloud in pixels |
| height | int | 338 | Height of the wordcloud in pixels |
| font_path | str | "fonts/Arial Unicode.ttf" | Path to the font file |
| margin | int | 2 | Margin between words in pixels |
| max_words | int | 200 | Maximum number of words to include |
| min_word_length | int | 3 | Minimum length of words to include |
| min_font_size | int | 14 | Minimum font size in points |
| max_font_size | int | None | Maximum font size in points |
| font_step | int | 2 | Step size for decreasing font size |
| stopwords | list | [] | Words to exclude from the wordcloud |
| background_color | str | 'white' | Background color |
| mode | str | "RGB" | Color mode ('RGB', 'RGBA', etc.) |
| black_white | bool | False | Whether to use only black text |
| place_strategy | str | 'random' | Strategy for word placement |
| rect_only | bool | False | Draw rectangles instead of text |
| tracing_files | bool | False | Generate debug tracing files |

#### Methods

- `generate(text_to_analyze)`: Generate a wordcloud from text
- `draw_image(save_file=False, image_name="wordcloud")`: Draw and optionally save the wordcloud as a PNG
- `draw_plt_image()`: Display the wordcloud using matplotlib
- `generate_svg(save_file=False, file_name="svg_img")`: Generate SVG representation
- `create_html(svg_content, save_file=False, file_name="wordcloud")`: Create interactive HTML with tooltips
- `update_position(new_fonts)`: Update word positions, optionally with new fonts
- `update_colors(new_colors)`: Update word colors using list or dictionary

## Advanced Customization

### Using Context Managers

```python
# Use the wordcloud as a context manager
with Wordcloud(width=800, height=400) as wc:
    wc.generate("Text for the wordcloud...")
    wc.draw_image(save_file=True)
```

### Updating Colors

```python
# Create basic wordcloud
wc = Wordcloud()
wc.generate("Text for the wordcloud...")

# Update colors with a list (positions-based)
new_colors = ["red", "blue", "green", "#FF5733", "rgb(100, 150, 200)"]
wc.update_colors(new_colors)

# Or update colors with a dictionary (word-based)
word_colors = {
    "important": "red",
    "python": "blue",
    "data": "green"
}
wc.update_colors(word_colors)
```

### Updating Positions with New Fonts

```python
# Create a wordcloud
wc = Wordcloud()
wc.generate("Text for the wordcloud...")

# Update positions with new fonts
new_fonts = ["fonts/OpenSans-Bold.ttf", "fonts/Roboto-Regular.ttf"]
wc.update_position(new_fonts)
```

## About This Project

This wordcloud generator was developed as part of a Master Thesis project focused on optimizing word cloud generation algorithms and visual representation techniques.

## License

MIT License - Copyright (c) Jan Blaha

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.