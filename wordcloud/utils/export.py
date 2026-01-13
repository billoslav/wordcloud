#!/usr/bin/env python3
"""
Export utilities for saving wordclouds to PNG and basic HTML.
Lightweight export utilities; keeps optional dependencies optional.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple, BinaryIO
import logging
import io
import html
import time

from .logging_config import get_logger
from .helpers import ensure_parent_dir, create_folder

try:
    from reportlab.pdfgen import canvas  # type: ignore
    from reportlab.lib.pagesizes import A4, A3, letter, legal, tabloid, PAGE_SIZES  # type: ignore
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.utils import ImageReader  # type: ignore
    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    REPORTLAB_AVAILABLE = False
    canvas = None  # type: ignore
    A4 = None  # type: ignore
    PAGE_SIZES = {}  # type: ignore
    colors = None  # type: ignore
    ImageReader = None  # type: ignore

try:
    from PIL import Image, ImageFont
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PIL_AVAILABLE = False
    Image = None
    ImageFont = None

logger = get_logger(__name__)


def export_image(image: "Image.Image", output_path: Union[str, Path], optimize: bool = True) -> None:
    """
    Save a PIL Image object to a PNG file.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow required for image export. Install with 'pip install Pillow'.")

    output_path = ensure_parent_dir(output_path)
    image.save(output_path, format="PNG", optimize=optimize)
    logger.info("Image saved to %s", output_path)


def export_webp(image: "Image.Image", output_path: Union[str, Path], 
                quality: int = 80, method: int = 6, lossless: bool = False) -> None:
    """
    Export image as WebP format.
    
    Args:
        image: PIL Image object
        output_path: Output file path
        quality: Quality setting (0-100, higher is better)
        method: Compression method (0-6, higher is slower but better compression)
        lossless: Whether to use lossless compression
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow required for WebP export. Install with 'pip install Pillow'.")
    
    try:
        output_path = ensure_parent_dir(output_path)
        save_kwargs = {
            "format": "WEBP",
            "quality": quality,
            "method": method,
        }
        if lossless:
            save_kwargs["lossless"] = True
        
        image.save(output_path, **save_kwargs)
        logger.info("WebP image saved to %s", output_path)
    except Exception as e:
        logger.error(f"Error saving WebP: {e}")
        raise IOError(f"Failed to save WebP: {e}")


def export_eps(wordcloud_instance, output_path: Union[str, Path]) -> None:
    """
    Export wordcloud as EPS (Encapsulated PostScript) format.
    
    Args:
        wordcloud_instance: Wordcloud instance with gen_positions
        output_path: Output file path
    """
    if not wordcloud_instance.gen_positions:
        raise ValueError("Wordcloud has no generated positions. Call generate() first.")
    
    try:
        output_path = ensure_parent_dir(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # EPS header
            f.write("%!PS-Adobe-3.0 EPSF-3.0\n")
            f.write(f"%%BoundingBox: 0 0 {wordcloud_instance.width} {wordcloud_instance.height}\n")
            f.write(f"%%Creator: Wordcloud Library\n")
            f.write(f"%%Title: Wordcloud\n")
            f.write(f"%%EndComments\n\n")
            
            # Set background color
            bg_color = wordcloud_instance.background_color
            if bg_color and bg_color != 'transparent':
                # Convert color to RGB values
                if bg_color.startswith('#'):
                    r = int(bg_color[1:3], 16) / 255.0
                    g = int(bg_color[3:5], 16) / 255.0
                    b = int(bg_color[5:7], 16) / 255.0
                else:
                    # Default to white
                    r, g, b = 1.0, 1.0, 1.0
                
                f.write(f"{r} {g} {b} setrgbcolor\n")
                f.write(f"0 0 {wordcloud_instance.width} {wordcloud_instance.height} rectfill\n\n")
            
            # Draw words
            for (word, freq, count), font_path, font_size, (x, y), orientation, color in wordcloud_instance.gen_positions:
                # Parse color
                if color.startswith('#'):
                    r = int(color[1:3], 16) / 255.0
                    g = int(color[3:5], 16) / 255.0
                    b = int(color[5:7], 16) / 255.0
                elif color.startswith('rgb('):
                    # Parse rgb(r, g, b)
                    rgb = color[4:-1].split(',')
                    r = int(rgb[0].strip()) / 255.0
                    g = int(rgb[1].strip()) / 255.0
                    b = int(rgb[2].strip()) / 255.0
                else:
                    r, g, b = 0.0, 0.0, 0.0  # Default to black
                
                # Set color
                f.write(f"{r} {g} {b} setrgbcolor\n")
                
                # Set font and size (simplified - EPS font handling is complex)
                font_name = font_path.split('/')[-1].replace('.ttf', '').replace(' ', '')
                f.write(f"/{font_name} findfont {font_size} scalefont setfont\n")
                
                # Handle rotation
                if orientation:
                    f.write(f"{x} {wordcloud_instance.height - y} translate\n")
                    f.write(f"{orientation} rotate\n")
                    f.write(f"0 0 moveto\n")
                else:
                    f.write(f"{x} {wordcloud_instance.height - y} moveto\n")
                
                # Draw text (escape special characters)
                escaped_word = word.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
                f.write(f"({escaped_word}) show\n")
                
                # Reset transformation if rotated
                if orientation:
                    f.write(f"{-orientation} rotate\n")
                    f.write(f"{-x} {-(wordcloud_instance.height - y)} translate\n")
                
                f.write("\n")
            
            f.write("showpage\n")
            f.write("%%EOF\n")
        
        logger.info("EPS file saved to %s", output_path)
    except Exception as e:
        logger.error(f"Error saving EPS: {e}")
        raise IOError(f"Failed to save EPS: {e}")

class PDFExporter:
    """
    Exports wordclouds as PDF documents.
    
    This class provides methods for creating PDF documents containing
    wordclouds, optionally with additional metadata and formatting.
    """
    
    def __init__(self):
        """
        Initialize the PDF exporter.
        """
        self.logger = logging.getLogger(__name__ + '.PDFExporter')
        
        if not REPORTLAB_AVAILABLE:
            self.logger.warning("ReportLab not available, PDF export functionality will be limited")
    
    def export_pdf(self, 
                 image: Image.Image, 
                 output: Union[str, Path, BinaryIO],
                 title: str = "WordCloud Plus Export",
                 author: str = "WordCloud Plus",
                 subject: str = "Word Cloud Visualization",
                 page_size: str = 'a4',
                 margin: int = 50,
                 include_metadata: bool = True,
                 include_timestamp: bool = True,
                 include_stats: bool = False,
                 stats_data: Optional[Dict[str, Any]] = None,
                 show_border: bool = False) -> None:
        """
        Export a wordcloud as a PDF document.
        
        Args:
            image: The wordcloud image to export
            output: Output file path or file-like object
            title: Document title
            author: Document author
            subject: Document subject
            page_size: Page size (a4, a3, letter, legal, tabloid)
            margin: Margin size in points
            include_metadata: Whether to include title, timestamp, etc.
            include_timestamp: Whether to include a timestamp
            include_stats: Whether to include statistics about the wordcloud
            stats_data: Optional statistics data to include
            show_border: Whether to show a border around the wordcloud
            
        Raises:
            RuntimeError: If ReportLab is not available
            IOError: If there's an error writing the PDF
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab is required for PDF export. Install with 'pip install reportlab'.")
            
        # Determine page size
        page_size_val = PAGE_SIZES.get(page_size.lower(), A4)
        page_width, page_height = page_size_val
            
        # Calculate available area
        content_width = page_width - 2 * margin
        content_height = page_height - 2 * margin
            
        # Create PDF canvas
        if isinstance(output, (str, Path)):
            output_path = ensure_parent_dir(output)
            c = canvas.Canvas(str(output_path), pagesize=page_size_val)
        else:
            c = canvas.Canvas(output, pagesize=page_size_val)
            
        try:
            # Set document properties
            if include_metadata:
                c.setTitle(title)
                c.setAuthor(author)
                c.setSubject(subject)
                
            # Calculate image placement
            img_width, img_height = image.size
            
            # Scale image to fit available area while maintaining aspect ratio
            width_ratio = content_width / img_width
            height_ratio = content_height / img_height
            scale_factor = min(width_ratio, height_ratio)
            
            scaled_width = img_width * scale_factor
            scaled_height = img_height * scale_factor
            
            # Center image on page
            x_position = margin + (content_width - scaled_width) / 2
            y_position = page_height - margin - scaled_height - (content_height - scaled_height) / 2
            
            # Draw border if requested
            if show_border:
                c.setStrokeColor(colors.black)
                c.setLineWidth(1)
                c.rect(x_position - 5, y_position - 5, 
                      scaled_width + 10, scaled_height + 10, 
                      stroke=1, fill=0)
            
            # Add the image
            img_data = io.BytesIO()
            image.save(img_data, format='PNG')
            img_data.seek(0)
            c.drawImage(ImageReader(img_data), x_position, y_position, 
                       width=scaled_width, height=scaled_height)
            
            # Add metadata at the bottom if requested
            if include_metadata:
                c.setFont("Helvetica", 12)
                c.drawString(margin, margin - 20, title)
                
                if include_timestamp:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    c.drawRightString(page_width - margin, margin - 20, f"Generated: {timestamp}")
            
            # Add statistics if requested
            if include_stats and stats_data:
                c.setFont("Helvetica", 10)
                y_pos = margin - 40
                for key, value in stats_data.items():
                    if y_pos < 20:  # Don't go below the page
                        break
                    c.drawString(margin, y_pos, f"{key}: {value}")
                    y_pos -= 15
            
            # Save the PDF
            c.save()
            
        except Exception as e:
            self.logger.error(f"Error creating PDF: {e}")
            raise IOError(f"Failed to create PDF: {e}")

class AnimatedGIFExporter:
    """
    Creates animated GIF wordclouds.
    
    This class provides methods for creating animated GIFs from a sequence of
    wordcloud frames, with various animation effects and transitions.
    """
    
    def __init__(self):
        """
        Initialize the animated GIF exporter.
        """
        self.logger = logging.getLogger(__name__ + '.AnimatedGIFExporter')
        
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available, animated GIF functionality will not work")
    
    def export_animated_gif(self, 
                          frames: List[Image.Image],
                          output: Union[str, Path, BinaryIO],
                          duration: int = 100,
                          loop: int = 0,
                          optimize: bool = True,
                          disposal: Optional[int] = None) -> None:
        """
        Export a sequence of frames as an animated GIF.
        
        Args:
            frames: List of images to use as frames
            output: Output file path or file-like object
            duration: Duration of each frame in milliseconds
            loop: Number of times to loop (0 = infinite)
            optimize: Whether to optimize the GIF
            disposal: Disposal method (how a frame is treated after display)
            
        Raises:
            RuntimeError: If PIL is not available
            IOError: If there's an error writing the GIF
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL is required for animated GIF export.")
            
        if not frames:
            raise ValueError("No frames provided for animation")
            
        try:
            save_kwargs = {
                "format": "GIF",
                "append_images": frames[1:],
                "save_all": True,
                "duration": duration,
                "loop": loop,
                "optimize": optimize,
            }
            # Pillow expects an int for disposal; do not pass None (it will crash)
            if disposal is not None:
                save_kwargs["disposal"] = disposal

            if isinstance(output, (str, Path)):
                output_path = ensure_parent_dir(output)
                
                # Save the GIF
                frames[0].save(str(output_path), **save_kwargs)
            else:
                # Save to file-like object
                frames[0].save(output, **save_kwargs)
                
        except Exception as e:
            self.logger.error(f"Error creating animated GIF: {e}")
            raise IOError(f"Failed to create animated GIF: {e}")
    
    def create_fade_in_animation(self, 
                              image: Image.Image, 
                              n_frames: int = 20,
                              background_color: Union[str, Tuple[int, int, int]] = "white") -> List[Image.Image]:
        """
        Create a fade-in animation for a wordcloud.
        
        Args:
            image: The final wordcloud image
            n_frames: Number of frames in the animation
            background_color: Background color to fade from
            
        Returns:
            List of animation frames
        """
        frames = []
        img_width, img_height = image.size
        
        # Convert string color to RGB if needed
        if isinstance(background_color, str):
            bg_img = Image.new('RGB', (1, 1), color=background_color)
            background_color = bg_img.getpixel((0, 0))
        
        for i in range(n_frames):
            # Calculate opacity for this frame
            opacity = i / (n_frames - 1)
            
            # Create a new frame
            frame = Image.new('RGBA', (img_width, img_height), color=(*background_color, 255))
            
            # Make a copy of the original with appropriate opacity
            img_opacity = image.copy().convert('RGBA')
            
            # Apply opacity to non-background pixels
            pixels = img_opacity.load()
            for y in range(img_height):
                for x in range(img_width):
                    r, g, b, a = pixels[x, y]
                    # Only apply opacity to non-background pixels
                    if (r, g, b, a) != (*background_color, 255):
                        pixels[x, y] = (r, g, b, int(a * opacity))
            
            # Composite the images
            frame.paste(img_opacity, (0, 0), img_opacity)
            frames.append(frame.convert('RGB'))
        
        return frames
    
    def create_grow_animation(self, 
                           image: Image.Image, 
                           n_frames: int = 20,
                           background_color: Union[str, Tuple[int, int, int]] = "white") -> List[Image.Image]:
        """
        Create an animation where the wordcloud grows from the center.
        
        Args:
            image: The final wordcloud image
            n_frames: Number of frames in the animation
            background_color: Background color
            
        Returns:
            List of animation frames
        """
        frames = []
        img_width, img_height = image.size
        
        for i in range(n_frames):
            # Calculate scale for this frame
            scale = 0.1 + 0.9 * (i / (n_frames - 1))
            
            # Create a new frame with background color
            frame = Image.new('RGB', (img_width, img_height), color=background_color)
            
            # Calculate dimensions of the scaled image
            scaled_width = int(img_width * scale)
            scaled_height = int(img_height * scale)
            
            # Resize the image
            scaled_img = image.resize((scaled_width, scaled_height), Image.LANCZOS)
            
            # Calculate position to center the scaled image
            x_offset = (img_width - scaled_width) // 2
            y_offset = (img_height - scaled_height) // 2
            
            # Paste the scaled image onto the frame
            frame.paste(scaled_img, (x_offset, y_offset))
            frames.append(frame)
        
        return frames
    
    def create_word_by_word_animation(self, 
                                   image: Image.Image, 
                                   word_positions: List[Tuple[str, int, int, int, int]],
                                   frames_per_word: int = 3,
                                   background_color: Union[str, Tuple[int, int, int]] = "white") -> List[Image.Image]:
        """
        Create an animation where words appear one by one.
        
        Args:
            image: The final wordcloud image (used for size only)
            word_positions: List of (word, x, y, width, height) tuples
            frames_per_word: Number of frames to show each new word
            background_color: Background color
            
        Returns:
            List of animation frames
        """
        frames = []
        img_width, img_height = image.size
        
        # Create an initial empty frame
        current_frame = Image.new('RGB', (img_width, img_height), color=background_color)
        frames.append(current_frame.copy())
        
        # Add words one by one
        for word_info in word_positions:
            word, x, y, width, height, draw_func = word_info
            
            # Make several copies of the frame with the new word
            for _ in range(frames_per_word):
                # Create a new frame by copying the previous one
                new_frame = current_frame.copy()
                
                # Draw the word using the provided drawing function
                draw_func(new_frame)
                
                # Add the frame to the list
                frames.append(new_frame.copy())
                
                # Update the current frame
                current_frame = new_frame
        
        return frames

class SVGExporter:
    """
    Exports wordclouds as SVG documents.
    
    This class provides methods for creating SVG documents containing
    wordclouds with various options for formatting and interactivity.
    """
    
    def __init__(self):
        """
        Initialize the SVG exporter.
        """
        self.logger = logging.getLogger(__name__ + '.SVGExporter')
    
    def export_svg(self, 
                 svg_content: str,
                 output: Union[str, Path, BinaryIO],
                 minify: bool = False) -> None:
        """
        Export a wordcloud as an SVG document.
        
        Args:
            svg_content: The SVG content as a string
            output: Output file path or file-like object
            minify: Whether to minify the SVG
            
        Raises:
            IOError: If there's an error writing the SVG
        """
        if minify:
            # Basic minification: remove whitespace and comments
            svg_content = self._minify_svg(svg_content)
        
        try:
            if isinstance(output, (str, Path)):
                # Use helper function to create folder
                output_path = Path(output)
                create_folder(output_path.parent)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
            else:
                # Write to file-like object
                if hasattr(output, 'write'):
                    # Encode if writing to a binary stream
                    output.write(svg_content.encode('utf-8') if isinstance(output, io.BufferedIOBase) else svg_content)
                else:
                    raise ValueError("Output must be a file path or a file-like object")
            
            logger.info(f"SVG saved to {output}")

        except Exception as e:
            self.logger.error(f"Error creating SVG: {e}")
            raise IOError(f"Failed to create SVG: {e}")
    
    def _minify_svg(self, svg_content: str) -> str:
        """
        Basic SVG minification: remove comments and excessive whitespace.
        
        Args:
            svg_content: The SVG content to minify
            
        Returns:
            Minified SVG content
        """
        import re
        
        # Remove comments
        svg_content = re.sub(r'<!--.*?-->', '', svg_content, flags=re.DOTALL)
        
        # Remove excessive whitespace
        svg_content = re.sub(r'\s+', ' ', svg_content)
        
        # Remove space between tags
        svg_content = re.sub(r'>\s+<', '><', svg_content)
        
        # Trim leading/trailing whitespace
        svg_content = svg_content.strip()
        
        return svg_content

    def generate_svg_content(self, 
                             gen_positions: List[Tuple[Any, str, int, Tuple[int, int], Optional[int], str]],
                             width: int, height: int, 
                             default_font_path: str, 
                             max_font_size: int,
                             background_color: Optional[str] = None) -> str:
        """
        Generates the SVG content string from word cloud data.

        Args:
            gen_positions: List containing word data and placement info 
                           (e.g., from Wordcloud.gen_positions).
            width: Width of the SVG canvas.
            height: Height of the SVG canvas.
            default_font_path: Default font path used for fallback/style info.
            max_font_size: Max font size used (for font info fallback).
            background_color: Optional background color for the SVG.

        Returns:
            The SVG content as a string.
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL/Pillow is required for font metrics needed for SVG generation.")

        result = []

        # Get font information (using default font path for general style)
        try:
            if not PIL_AVAILABLE or ImageFont is None:
                raise RuntimeError("PIL/Pillow is required for font metrics needed for SVG generation.")
            # Use a reasonable size for font name fetching, e.g., 10
            font_for_name = ImageFont.truetype(default_font_path, 10) 
            raw_font_family, raw_font_style = font_for_name.getname()
            font_family = repr(raw_font_family)
            raw_font_style = raw_font_style.lower()
            # Basic style interpretation (can be improved)
            font_style = 'italic' if 'italic' in raw_font_style else 'normal'
            font_weight = 'bold' if 'bold' in raw_font_style else 'normal'
        except Exception as e:
            logger.warning(f"Could not load font {default_font_path} for SVG style info: {e}. Using defaults.")
            font_family = "sans-serif"
            font_style = "normal"
            font_weight = "normal"

        # Header
        result.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')

        # Style
        result.append(f'<style>text{{font-family:{font_family}; font-weight:{font_weight}; font-style:{font_style};}}</style>')

        # Background if defined
        if background_color:
            result.append(f'<rect width="100%" height="100%" style="fill:{background_color}"></rect>')

        # Add text elements
        for (word, freq, count), font_path, font_size, (x, y), orientation, color in gen_positions:
            try:
                if not PIL_AVAILABLE or ImageFont is None:
                    raise RuntimeError("PIL/Pillow is required for font metrics needed for SVG generation.")
                font = ImageFont.truetype(font_path, font_size)
                # Get bounding box and metrics to adjust position for SVG baseline
                # Note: PIL's bbox might not perfectly align with SVG text rendering.
                bbox = font.getbbox(word) # (left, top, right, bottom)
                if bbox is None: # Handle cases where font fails to get bbox
                    logger.warning(f"Could not get bounding box for word '{word}' with font {font_path} size {font_size}")
                    continue 
                ascent, descent = font.getmetrics()

                # Adjust coordinates: PIL draws relative to top-left, SVG uses baseline
                svg_x = x - bbox[0]
                svg_y = y + (ascent - bbox[1]) # Adjust y based on ascent and bbox top

                # Handle rotation (optional - basic implementation)
                transform = f"translate({svg_x},{svg_y})"
                if orientation and orientation != 0:
                    # SVG rotation is around (0,0) of the translated coordinate system.
                    transform += f" rotate({orientation})" 
                    # TODO: Implement rotation around word center if needed.

                # Escape special XML characters in the word itself
                escaped_word = word.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                # Escape quotes in the ID attribute
                escaped_id = word.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")

                # Create text element
                result.append(f'<text id="{escaped_id}" transform="{transform}" font-size="{font_size}" style="fill:{color}" count="{count}">{escaped_word}</text>')
            except Exception as e:
                logger.error(f"Error processing word '{word}' for SVG: {e}")

        result.append('</svg>')
        return '\n'.join(result)
    
    def generate_and_export_svg(self, 
                                gen_positions: List[Tuple[Any, str, int, Tuple[int, int], Optional[int], str]],
                                width: int, height: int, 
                                default_font_path: str, 
                                max_font_size: int,
                                output: Union[str, Path, BinaryIO],
                                background_color: Optional[str] = None,
                                minify: bool = False) -> None:
        """
        Generates SVG content from word cloud data and exports it to a file.

        Args:
            gen_positions: List containing word data and placement info.
            width: Width of the SVG canvas.
            height: Height of the SVG canvas.
            default_font_path: Default font path used.
            max_font_size: Max font size used (or a default if None).
            output: Output file path or file-like object.
            background_color: Optional background color.
            minify: Whether to minify the SVG.
        """
        # Use a default max_font_size if None is passed, e.g., for getname()
        effective_max_font_size = max_font_size if max_font_size is not None else 80 
        
        svg_content = self.generate_svg_content(
            gen_positions, width, height, default_font_path, 
            effective_max_font_size, background_color
        )
        self.export_svg(svg_content, output, minify=minify)

# Default instances
_pdf_exporter = None
_gif_exporter = None
_svg_exporter = None

def get_pdf_exporter() -> PDFExporter:
    """
    Get or create the global PDFExporter instance.
    
    Returns:
        A PDFExporter instance
    """
    global _pdf_exporter
    if _pdf_exporter is None:
        _pdf_exporter = PDFExporter()
    return _pdf_exporter

def get_gif_exporter() -> AnimatedGIFExporter:
    """
    Get or create the global AnimatedGIFExporter instance.
    
    Returns:
        An AnimatedGIFExporter instance
    """
    global _gif_exporter
    if _gif_exporter is None:
        _gif_exporter = AnimatedGIFExporter()
    return _gif_exporter

def get_svg_exporter() -> SVGExporter:
    """
    Get or create the global SVGExporter instance.
    
    Returns:
        An SVGExporter instance
    """
    global _svg_exporter
    if _svg_exporter is None:
        _svg_exporter = SVGExporter()
    return _svg_exporter

# --- Simple HTML Export ---

def export_simple_html(
    words_data: List[Tuple[Tuple[str, float, int], str, float, Tuple[int, int], Optional[str], str]],
    output_file: Optional[str] = None,
    width: int = 400,
    height: int = 200,
    background_color: str = "white",
    css_file: Optional[str] = None
) -> str:
    """Generates a simple HTML representation of the word cloud and returns it as a string.

    Optionally saves the HTML to a file.

    Args:
        words_data: List containing placement data for each word.
                    Format: [( (word, freq, count), font_path, font_size, position, orientation, color), ...]
        output_file: Optional path to save the generated HTML file.
        width: Width of the cloud container.
        height: Height of the cloud container.
        background_color: Background color for the container.
        css_file: Optional path to an external CSS file to link.

    Returns:
        The generated HTML content as a string.
    """
    html_content = []
    html_content.append("<!DOCTYPE html>")
    html_content.append("<html>")
    html_content.append("<head>")
    html_content.append('<meta charset="UTF-8">')
    html_content.append("<title>Word Cloud</title>")
    if css_file:
        html_content.append(f'<link rel="stylesheet" href="{css_file}">')
    html_content.append("<style>")
    html_content.append("  .wordcloud-container {")
    html_content.append(f"    width: {width}px;")
    html_content.append(f"    height: {height}px;")
    html_content.append(f"    background-color: {background_color};")
    html_content.append("    position: relative; /* Crucial for absolute positioning of words */")
    html_content.append("    border: 1px solid #ccc; /* Optional border */")
    html_content.append("    overflow: hidden; /* Hide overflow if words go outside bounds */")
    html_content.append("  }")
    html_content.append("  .wordcloud-word {")
    html_content.append("    position: absolute;")
    html_content.append("    box-sizing: border-box; /* Include padding/border in element's total width/height */")
    # Basic font styling - can be overridden by inline styles or external CSS
    html_content.append("    font-family: sans-serif; ")
    html_content.append("    white-space: nowrap; /* Prevent words from wrapping */")
    html_content.append("  }")
    html_content.append("</style>")
    html_content.append("</head>")
    html_content.append("<body>")
    html_content.append(f'<div class="wordcloud-container" style="width:{width}px; height:{height}px; background-color:{background_color};">')

    for word_info in words_data:
        (word, _, _), font_path, font_size, position, orientation, color = word_info
        escaped_word = html.escape(word)
        pos_x, pos_y = position

        # Basic style application - font family might need more sophisticated handling
        # if font_path is complex. Rotation via CSS transform if orientation is present.
        style = f"left: {pos_x}px; top: {pos_y}px; font-size: {font_size}px; color: {color};"
        # TODO: Map font_path to a web-safe font-family or use @font-face if needed
        # style += f" font-family: '{Path(font_path).stem}';" # Simple example, likely needs improvement


        transform = ""
        if orientation == "horizontal": # Explicit horizontal (or default)
             pass
        elif orientation == "vertical":
            # CSS rotation for vertical text. Might need adjustments for alignment.
            # transform = "transform: rotate(90deg) translate(0, -100%); transform-origin: top left;"
            # Simpler rotation for now, might clip text:
             transform = "transform: rotate(90deg); transform-origin: center center;" # Centered rotation might be visually better
             # TODO: Vertical text positioning in HTML/CSS is tricky. Needs refinement.
        elif orientation: # Handle other potential orientation values if defined
             # Example: Add rotations based on specific orientation tags if needed
             pass


        html_content.append(f'  <span class="wordcloud-word" style="{style} {transform}" title="{escaped_word} ({font_size}px)">')
        html_content.append(f"    {escaped_word}")
        html_content.append("  </span>")

    html_content.append("</div>")
    html_content.append("</body>")
    html_content.append("</html>")

    final_html = "\n".join(html_content)

    # Save to file if requested
    if output_file:
        try:
            output_path = ensure_parent_dir(output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_html)
            logger.info(f"Simple HTML word cloud saved to: {output_path.resolve()}")
        except Exception as e:
            logger.error(f"Error saving simple HTML to {output_file}: {e}")

    return final_html

def export_interactive_html(wc: Any, output_path: Union[str, Path]) -> None:
    """
    Generates and saves an advanced interactive HTML visualization.

    Args:
        wc: The Wordcloud instance containing generated data.
        output_path: The path to save the HTML file.

    Raises:
        RuntimeError: If required data (gen_positions) is missing.
        IOError: If there's an error saving the HTML file.
    """
    if not wc.gen_positions:
        raise RuntimeError("Cannot generate interactive HTML: Wordcloud positions not generated yet.")

    # Generate base SVG content using the utility function
    svg_exporter = get_svg_exporter()
    svg_content = svg_exporter.generate_svg_content(
        gen_positions=wc.gen_positions,
        width=wc.width, 
        height=wc.height,
        default_font_path=wc.font_path, # Use the instance's font path
        max_font_size=wc.max_font_size or wc.def_max_font_size, 
        background_color=wc.background_color
    )

    # --- Start building the interactive HTML --- 
    html_content = []
    
    # Header and complex CSS (adapted from Wordcloud.generate_interactive_html)
    html_content.append("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Interactive Wordcloud</title>
  <style>
    :root {
      --background-color: #f8f9fa;
      --text-color: #333;
      --panel-bg-color: rgba(255, 255, 255, 0.95);
      --panel-border-color: #ddd;
      --primary-color: #3498db;
      --secondary-color: #2ecc71;
      --highlight-color: #f39c12;
      --font-family: "Arial", sans-serif; /* Fallback font */
    }
    body {
      font-family: var(--font-family);
      margin: 0;
      padding: 20px;
      background-color: var(--background-color);
      color: var(--text-color);
      display: flex;
      gap: 20px;
      min-height: 100vh;
      box-sizing: border-box;
    }
    .wordcloud-container {
      flex-grow: 1;
      position: relative; /* Needed for absolute positioning of info panel */
    }
    .controls-panel {
      width: 300px;
      flex-shrink: 0;
      background-color: var(--panel-bg-color);
      border: 1px solid var(--panel-border-color);
      border-radius: 8px;
      padding: 15px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      max-height: calc(100vh - 40px);
      display: flex;
      flex-direction: column;
    }
    .controls-panel h2 {
      margin-top: 0;
      font-size: 1.2em;
      border-bottom: 1px solid var(--panel-border-color);
      padding-bottom: 10px;
      margin-bottom: 15px;
      color: var(--primary-color);
    }
    .stats-section, .controls-section, .word-list-section {
      margin-bottom: 20px;
    }
    .stats-section div {
      margin-bottom: 8px;
      font-size: 0.9em;
    }
    .stats-section span:first-child {
        font-weight: normal;
        color: var(--text-color);
        display: inline-block;
        min-width: 100px;
    }
    .stats-section span:last-child {
        font-weight: bold;
        color: var(--secondary-color);
    }
    .controls-section button, .controls-section select {
      display: block;
      width: 100%;
      padding: 8px 12px;
      margin-bottom: 10px;
      border: 1px solid var(--panel-border-color);
      border-radius: 4px;
      background-color: #fff;
      cursor: pointer;
      transition: background-color 0.2s ease;
    }
    .controls-section button:hover {
      background-color: #eee;
    }
    #word-list-container {
        max-height: 300px; /* Limit height and make scrollable */
        overflow-y: auto;
        border: 1px solid var(--panel-border-color);
        padding: 5px;
        border-radius: 4px;
    }
    .word-item {
        display: flex;
        justify-content: space-between;
        padding: 4px 8px;
        font-size: 0.85em;
        cursor: pointer;
        border-radius: 3px;
    }
    .word-item:hover {
        background-color: #eee;
    }
    .word-item span:last-child {
        font-weight: bold;
        color: var(--primary-color);
    }
    #info-panel {
      position: absolute;
      background-color: var(--panel-bg-color);
      border: 1px solid var(--panel-border-color);
      border-radius: 5px;
      padding: 10px 15px;
      font-size: 0.9em;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      display: none;
      z-index: 100;
      pointer-events: none; /* Allow clicks through */
    }
    #info-panel h3 {
        margin: 0 0 5px 0;
        font-size: 1.1em;
        color: var(--primary-color);
    }
    svg text {
      cursor: default;
      transition: transform 0.2s ease-out, opacity 0.2s ease-out;
      animation: fadeIn 0.5s ease-out forwards;
      opacity: 0; /* Start hidden for animation */
    }
    svg text:hover {
      transform: scale(1.1);
      opacity: 1 !important; /* Ensure hover overrides dimming */
      fill: var(--highlight-color) !important; /* Highlight color */
    }
    @keyframes fadeIn {
      to { opacity: 1; }
    }
  </style>
</head>
<body>""")

    # Body Structure
    html_content.append("""
  <div class="wordcloud-container">
    <!-- SVG will be embedded here -->
    {svg_inner}
    <div id="info-panel">
      <h3 id="word-title">Word Info</h3>
      <div>Count: <span id="word-count"></span></div>
      <div>Frequency: <span id="word-freq"></span></div>
    </div>
  </div>

  <div class="controls-panel">
    <h2>Wordcloud Info & Controls</h2>
    
    <div class="stats-section">
      <h3>Statistics</h3>
      <div><span>Total Words:</span> <span id="total-words">N/A</span></div>
      <div><span>Unique Words:</span> <span id="unique-words">N/A</span></div>
      <div><span>Most Common:</span> <span id="most-common">N/A</span></div>
      <div><span>Largest Word:</span> <span id="largest-word">N/A</span></div>
    </div>
    
    <div class="word-list-section">
      <h3>Top Words</h3>
      <div id="word-list-container">
        <!-- Word list will be populated by JS -->
      </div>
    </div>
    
    <div class="controls-section">
      <h3>Controls</h3>
      <select id="theme-selector">
        <option value="default">Default Theme</option>
        <option value="dark">Dark Theme</option>
        <option value="colorful">Colorful Theme</option>
      </select>
      <button id="reset-btn">Reset View</button>
      <button id="download-btn">Download SVG</button>
      <!-- <button id="animate-btn">Re-Animate</button> -->
    </div>
    
  </div>
""")

    # Extract inner SVG content (remove outer <svg> tag)
    import re
    svg_match = re.search(r'<svg[^>]*>(.*?)</svg>', svg_content, re.DOTALL)
    if not svg_match:
        logger.error("Failed to extract SVG content for interactive HTML")
        svg_inner = svg_content # Fallback
    else:
        svg_inner = svg_match.group(1)

    # Insert SVG content into the body structure
    # Find the placeholder and replace it
    placeholder = "{svg_inner}"
    body_index = -1
    for i, line in enumerate(html_content):
        if placeholder in line:
            html_content[i] = line.replace(placeholder, svg_inner)
            body_index = i
            break
            
    if body_index == -1:
        logger.error("Could not find SVG placeholder in HTML template.")
        # Append SVG at the end as fallback
        html_content.insert(-1, svg_inner) 

    # JavaScript (adapted from Wordcloud.generate_interactive_html)
    js_content = []
    js_content.append("""<script>
    const wordData = {
""") # Start the first part of JS
    
    word_data_entries = []
    for (word, freq, count), font_path, font_size, position, orientation, color in wc.gen_positions:
        # Escape quotes and backslashes in word for JS string literal
        safe_word = word.replace('\\', '\\\\').replace('"', '\\"')
        word_data_entries.append(f'      "{safe_word}": {{ count: {count}, freq: {freq:.4f}, fontSize: {font_size}, color: "{color}" }}')
    
    js_content.append(',\n'.join(word_data_entries))
    
    # Append the rest of the JavaScript code
    js_content.append(""" 
    }; // Close wordData object
     
    // Calculate statistics
    const words = Object.keys(wordData);
    const uniqueWords = words.length;
    let totalCount = 0;
    let mostCommonWord = '';
    let mostCommonCount = 0;
    let largestWord = '';
    let largestSize = 0;
    
    words.forEach(word => {
      const data = wordData[word];
      totalCount += data.count;
      
      if (data.count > mostCommonCount) {
        mostCommonCount = data.count;
        mostCommonWord = word;
      }
      
      if (data.fontSize > largestSize) {
        largestSize = data.fontSize;
        largestWord = word;
      }
    });
    
    // Update statistics panel
    document.getElementById('total-words').textContent = totalCount;
    document.getElementById('unique-words').textContent = uniqueWords;
    document.getElementById('most-common').textContent = `${mostCommonWord} (${mostCommonCount})`;
    document.getElementById('largest-word').textContent = largestWord;
    
    // Populate word list (top N words)
    const wordListContainer = document.getElementById('word-list-container');
    const topWords = words
      .sort((a, b) => wordData[b].count - wordData[a].count)
      .slice(0, 20); // Show top 20
      
    topWords.forEach(word => {
      const div = document.createElement('div');
      div.className = 'word-item';
      // Escape HTML entities for display
      const escapedWord = word.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
      div.innerHTML = `
        <span>${escapedWord}</span>
        <span>${wordData[word].count}</span>
      `;
      div.addEventListener('click', () => {
        // Escape ID selector: CSS.escape() is ideal but not universally available
        // Basic escaping for common issues:
        const escapedId = word.replace(/([\"\'\\#\\.!<>\\(\\)\\[\\]])/g, '\\$1');
        try {
          const textEl = document.getElementById(escapedId);
          if (textEl) {
            // Simulate hover to show info panel
            textEl.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
            // Scroll word into view
            setTimeout(() => textEl.scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
          }
        } catch (e) {
            console.error("Could not find or interact with element for word:", word, "Escaped ID:", escapedId, e);
        }
      });
      wordListContainer.appendChild(div);
    });
    
    // Set up event listeners for words in SVG
    document.querySelectorAll('svg text').forEach(textElement => {
      const wordId = textElement.id; // ID might be escaped
      // Find the original word key in wordData matching the potentially escaped ID
      const word = Object.keys(wordData).find(key => 
           key.replace(/([\"\'\\#\\.!<>\\(\\)\\[\\]])/g, '\\$1') === wordId
      );
      
      if (!word || !wordData[word]) {
        console.warn("Could not link SVG text element to data:", textElement.id);
        return; // Skip if word data not found
      }
      const data = wordData[word];

      // Add animation delay based on word size/rank (optional)
      const delay = (words.indexOf(word) * 0.02).toFixed(2); // Simple delay based on rank
      textElement.style.animationDelay = `${delay}s`;
      
      // Hover effect
      textElement.addEventListener('mouseover', () => {
        // Update info panel
        const displayWord = word.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); // Escape for display
        document.getElementById('word-title').textContent = displayWord;
        document.getElementById('word-count').textContent = data.count;
        document.getElementById('word-freq').textContent = (data.freq * 100).toFixed(1) + '%';
        const infoPanel = document.getElementById('info-panel');
        infoPanel.style.display = 'block';
        
        // Position the info panel near the word
        const rect = textElement.getBoundingClientRect();
        const panelRect = infoPanel.getBoundingClientRect();
        
        let top = rect.top + window.scrollY;
        let left = rect.right + window.scrollX + 15; // Default to right

        // Adjust if panel goes off screen
        if (left + panelRect.width > window.innerWidth) { // Off right edge
          left = rect.left + window.scrollX - panelRect.width - 15;
        }
        if (top + panelRect.height > window.innerHeight) { // Off bottom edge
          top = rect.bottom + window.scrollY - panelRect.height;
        }
         if (left < 0) left = 10; // Prevent going off left edge
         if (top < 0) top = 10; // Prevent going off top edge
        
        infoPanel.style.left = `${left}px`;
        infoPanel.style.top = `${top}px`;
        
        // Highlight this word and dim others
        document.querySelectorAll('svg text').forEach(t => {
          if (t !== textElement) t.style.opacity = '0.3';
        });
        textElement.style.opacity = '1'; // Ensure hovered is fully visible
      });
      
      textElement.addEventListener('mouseout', () => {
        document.getElementById('info-panel').style.display = 'none';
        // Restore opacity of all words
        document.querySelectorAll('svg text').forEach(t => {
          t.style.opacity = '1';
        });
      });
    });
    
    // Control buttons functionality
    document.getElementById('reset-btn').addEventListener('click', () => {
      document.querySelectorAll('svg text').forEach(t => {
        t.style.transform = 'scale(1)';
        t.style.opacity = '1';
      });
      document.getElementById('info-panel').style.display = 'none';
    });
    
    // Theme selector functionality
    document.getElementById('theme-selector').addEventListener('change', (e) => {
      const theme = e.target.value;
      const root = document.documentElement;
      
      if (theme === 'dark') {
        root.style.setProperty('--background-color', '#222');
        root.style.setProperty('--text-color', '#eee');
        root.style.setProperty('--panel-bg-color', 'rgba(40, 40, 40, 0.95)');
        root.style.setProperty('--panel-border-color', '#444');
        root.style.setProperty('--primary-color', '#5dade2'); // Lighter blue
        root.style.setProperty('--secondary-color', '#58d68d'); // Lighter green
        root.style.setProperty('--highlight-color', '#f5b041'); // Lighter orange
        
        const svgRect = document.querySelector('svg rect');
        if (svgRect) svgRect.setAttribute('style', 'fill:#333');
      } else if (theme === 'colorful') {
        root.style.setProperty('--background-color', '#eaf2f8'); // Light blue bg
        root.style.setProperty('--text-color', '#17202a');
        root.style.setProperty('--panel-bg-color', 'rgba(255, 255, 255, 0.95)');
        root.style.setProperty('--panel-border-color', '#aed6f1');
        root.style.setProperty('--primary-color', '#9b59b6'); // Purple
        root.style.setProperty('--secondary-color', '#e74c3c'); // Red
        root.style.setProperty('--highlight-color', '#f1c40f'); // Yellow
        
        const svgRect = document.querySelector('svg rect');
        if (svgRect) svgRect.setAttribute('style', 'fill:#fdfefe');
      } else {
        // Default theme reset
        root.style.setProperty('--background-color', '#f8f9fa');
        root.style.setProperty('--text-color', '#333');
        root.style.setProperty('--panel-bg-color', 'rgba(255, 255, 255, 0.95)');
        root.style.setProperty('--panel-border-color', '#ddd');
        root.style.setProperty('--primary-color', '#3498db');
        root.style.setProperty('--secondary-color', '#2ecc71');
        root.style.setProperty('--highlight-color', '#f39c12');
        
        const svgRect = document.querySelector('svg rect');
        // Use the original background color for the default theme rectangle
        const originalBgColor = wordData[Object.keys(wordData)[0]] ? wc.background_color : 'white'; // Fetch from wc instance or default
        if (svgRect) svgRect.setAttribute('style', `fill:${originalBgColor || 'white'}`); 
      }
    });
    
    // Download functionality
    document.getElementById('download-btn').addEventListener('click', () => {
      const svgElement = document.querySelector('div.wordcloud-container svg');
      if (!svgElement) return;
      
      const serializer = new XMLSerializer();
      let source = serializer.serializeToString(svgElement);
      
      // Add name spaces if missing
      if(!source.match(/^<svg[^>]+xmlns="http:\\/\\/www\\.w3\\.org\\/2000\\/svg"/)) {
        source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
      }
      if(!source.match(/^<svg[^>]+xmlns:xlink="http:\\/\\/www\\.w3\\.org\\/1999\\/xlink"/)) {
        source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
      }
      
      // Add XML declaration
      source = '<?xml version="1.0" standalone="no"?>\r\n' + source;
      
      // Convert SVG source to URL data
      const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(source);
      
      // Create download link
      const downloadLink = document.createElement("a");
      downloadLink.href = url;
      downloadLink.download = "wordcloud.svg";
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
    });

    // Trigger initial theme application if not default
    document.getElementById('theme-selector').dispatchEvent(new Event('change'));

  </script>
</body>
</html>
""")

    html_content.append('\n'.join(js_content))
    
    # Save the final HTML
    output_path = Path(output_path)
    create_folder(output_path.parent)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(html_content))
        logger.info(f"Interactive HTML saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving interactive HTML file {output_path}: {e}")
        raise IOError(f"Failed to save interactive HTML: {e}") 