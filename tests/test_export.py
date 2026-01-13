"""
Tests for export utilities: PDF, GIF, SVG, and HTML exports.
"""

import os
import io
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test imports
from wordcloud.utils.export import (
    export_image,
    PDFExporter,
    AnimatedGIFExporter,
    SVGExporter,
    get_pdf_exporter,
    get_gif_exporter,
    get_svg_exporter,
    export_simple_html,
    REPORTLAB_AVAILABLE,
    PIL_AVAILABLE,
)
from wordcloud.utils.helpers import create_folder

# Only import PIL if available
if PIL_AVAILABLE:
    from PIL import Image


class TestCreateFolder:
    """Tests for the create_folder utility function."""
    
    def test_create_new_folder(self, tmp_path):
        """Test creating a new folder."""
        new_folder = tmp_path / "new_folder"
        assert not new_folder.exists()
        
        create_folder(new_folder)
        
        assert new_folder.exists()
        assert new_folder.is_dir()
    
    def test_create_existing_folder(self, tmp_path):
        """Test creating a folder that already exists."""
        existing_folder = tmp_path / "existing"
        existing_folder.mkdir()
        
        # Should not raise an error
        create_folder(existing_folder)
        assert existing_folder.exists()
    
    def test_create_nested_folders(self, tmp_path):
        """Test creating nested folders."""
        nested = tmp_path / "a" / "b" / "c"
        
        create_folder(nested)
        
        assert nested.exists()
        assert nested.is_dir()


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
class TestExportImage:
    """Tests for the export_image function."""
    
    def test_export_png(self, tmp_path):
        """Test exporting a PIL image to PNG."""
        img = Image.new("RGB", (100, 100), color="red")
        output_path = tmp_path / "test.png"
        
        export_image(img, output_path)
        
        assert output_path.exists()
        # Verify it's a valid image
        loaded = Image.open(output_path)
        assert loaded.size == (100, 100)
    
    def test_export_creates_parent_dirs(self, tmp_path):
        """Test that export_image creates parent directories."""
        img = Image.new("RGB", (50, 50), color="blue")
        output_path = tmp_path / "subdir" / "another" / "test.png"
        
        export_image(img, output_path)
        
        assert output_path.exists()


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
class TestSVGExporter:
    """Tests for the SVGExporter class."""
    
    def test_export_svg_to_file(self, tmp_path):
        """Test exporting SVG content to a file."""
        exporter = SVGExporter()
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40"/></svg>'
        output_path = tmp_path / "test.svg"
        
        exporter.export_svg(svg_content, output_path)
        
        assert output_path.exists()
        with open(output_path, 'r') as f:
            content = f.read()
        assert '<circle' in content
    
    def test_export_svg_minified(self, tmp_path):
        """Test exporting minified SVG."""
        exporter = SVGExporter()
        svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <!-- A comment -->
            <circle cx="50" cy="50" r="40"/>
        </svg>'''
        output_path = tmp_path / "test_minified.svg"
        
        exporter.export_svg(svg_content, output_path, minify=True)
        
        with open(output_path, 'r') as f:
            content = f.read()
        # Comments should be removed
        assert '<!-- A comment -->' not in content
        # Excessive whitespace should be removed
        assert '\n' not in content or content.count('\n') < 3
    
    def test_generate_svg_content(self):
        """Test generating SVG content from word positions."""
        exporter = SVGExporter()
        
        # Mock word positions
        gen_positions = [
            (("hello", 1.0, 10), "fonts/Arial.ttf", 24, (50, 50), None, "rgb(255,0,0)"),
            (("world", 0.5, 5), "fonts/Arial.ttf", 18, (100, 100), None, "rgb(0,255,0)"),
        ]
        
        with patch('wordcloud.utils.export.PIL_AVAILABLE', True):
            with patch('wordcloud.utils.export.ImageFont') as mock_font:
                # Mock font behavior
                mock_font_instance = MagicMock()
                mock_font_instance.getname.return_value = ("Arial", "Regular")
                mock_font_instance.getbbox.return_value = (0, 0, 50, 20)
                mock_font_instance.getmetrics.return_value = (16, 4)
                mock_font.truetype.return_value = mock_font_instance
                
                svg = exporter.generate_svg_content(
                    gen_positions, 
                    width=200, 
                    height=150,
                    default_font_path="fonts/Arial.ttf",
                    max_font_size=24,
                    background_color="white"
                )
        
        assert '<svg' in svg
        assert 'width="200"' in svg
        assert 'height="150"' in svg
        assert '</svg>' in svg
    
    def test_get_svg_exporter_singleton(self):
        """Test that get_svg_exporter returns a singleton."""
        exporter1 = get_svg_exporter()
        exporter2 = get_svg_exporter()
        
        assert exporter1 is exporter2


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
class TestAnimatedGIFExporter:
    """Tests for the AnimatedGIFExporter class."""
    
    def test_export_animated_gif(self, tmp_path):
        """Test creating an animated GIF from frames."""
        exporter = AnimatedGIFExporter()
        
        # Create simple frames
        frames = [
            Image.new("RGB", (50, 50), color="red"),
            Image.new("RGB", (50, 50), color="green"),
            Image.new("RGB", (50, 50), color="blue"),
        ]
        
        output_path = tmp_path / "test.gif"
        exporter.export_animated_gif(frames, output_path, duration=100)
        
        assert output_path.exists()
        # Verify it's a valid GIF
        loaded = Image.open(output_path)
        assert loaded.format == "GIF"
    
    def test_export_animated_gif_empty_frames_raises(self):
        """Test that exporting with no frames raises an error."""
        exporter = AnimatedGIFExporter()
        
        with pytest.raises(ValueError, match="No frames provided"):
            exporter.export_animated_gif([], "output.gif")
    
    def test_create_fade_in_animation(self):
        """Test creating a fade-in animation."""
        exporter = AnimatedGIFExporter()
        
        img = Image.new("RGBA", (100, 100), color="red")
        frames = exporter.create_fade_in_animation(img, n_frames=5)
        
        assert len(frames) == 5
        for frame in frames:
            assert frame.size == (100, 100)
    
    def test_create_grow_animation(self):
        """Test creating a grow animation."""
        exporter = AnimatedGIFExporter()
        
        img = Image.new("RGB", (100, 100), color="blue")
        frames = exporter.create_grow_animation(img, n_frames=5)
        
        assert len(frames) == 5
        for frame in frames:
            assert frame.size == (100, 100)
    
    def test_get_gif_exporter_singleton(self):
        """Test that get_gif_exporter returns a singleton."""
        exporter1 = get_gif_exporter()
        exporter2 = get_gif_exporter()
        
        assert exporter1 is exporter2


@pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
class TestPDFExporter:
    """Tests for the PDFExporter class."""
    
    def test_export_pdf(self, tmp_path):
        """Test exporting an image to PDF."""
        exporter = PDFExporter()
        
        img = Image.new("RGB", (200, 200), color="white")
        output_path = tmp_path / "test.pdf"
        
        exporter.export_pdf(img, output_path, title="Test PDF")
        
        assert output_path.exists()
        # Check file has content
        assert output_path.stat().st_size > 0
    
    def test_export_pdf_with_border(self, tmp_path):
        """Test exporting PDF with a border."""
        exporter = PDFExporter()
        
        img = Image.new("RGB", (200, 200), color="lightblue")
        output_path = tmp_path / "test_border.pdf"
        
        exporter.export_pdf(img, output_path, show_border=True)
        
        assert output_path.exists()
    
    def test_export_pdf_with_stats(self, tmp_path):
        """Test exporting PDF with statistics."""
        exporter = PDFExporter()
        
        img = Image.new("RGB", (200, 200), color="green")
        output_path = tmp_path / "test_stats.pdf"
        stats = {"Total Words": 100, "Unique Words": 50}
        
        exporter.export_pdf(img, output_path, include_stats=True, stats_data=stats)
        
        assert output_path.exists()
    
    def test_get_pdf_exporter_singleton(self):
        """Test that get_pdf_exporter returns a singleton."""
        exporter1 = get_pdf_exporter()
        exporter2 = get_pdf_exporter()
        
        assert exporter1 is exporter2


class TestExportSimpleHTML:
    """Tests for the export_simple_html function."""
    
    def test_generate_html(self):
        """Test generating simple HTML content."""
        words_data = [
            (("hello", 1.0, 10), "fonts/Arial.ttf", 24, (50, 50), None, "rgb(255,0,0)"),
            (("world", 0.5, 5), "fonts/Arial.ttf", 18, (100, 100), None, "rgb(0,255,0)"),
        ]
        
        html = export_simple_html(words_data, width=300, height=200)
        
        assert "<!DOCTYPE html>" in html
        assert "hello" in html
        assert "world" in html
        assert "wordcloud-container" in html
    
    def test_generate_html_with_file(self, tmp_path):
        """Test saving HTML to a file."""
        words_data = [
            (("test", 1.0, 5), "fonts/Arial.ttf", 20, (25, 25), None, "black"),
        ]
        output_file = tmp_path / "test.html"
        
        html = export_simple_html(words_data, output_file=str(output_file))
        
        assert output_file.exists()
        with open(output_file, 'r') as f:
            content = f.read()
        assert "test" in content
    
    def test_generate_html_with_css_file(self):
        """Test generating HTML with custom CSS file reference."""
        words_data = [
            (("word", 1.0, 5), "fonts/Arial.ttf", 16, (10, 10), None, "blue"),
        ]
        
        html = export_simple_html(words_data, css_file="custom.css")
        
        assert 'href="custom.css"' in html
    
    def test_generate_html_background_color(self):
        """Test generating HTML with custom background color."""
        words_data = [
            (("word", 1.0, 5), "fonts/Arial.ttf", 16, (10, 10), None, "blue"),
        ]
        
        html = export_simple_html(words_data, background_color="#333333")
        
        assert "#333333" in html


class TestExporterWithMocks:
    """Tests using mocks for unavailable dependencies."""
    
    def test_pdf_exporter_without_reportlab(self):
        """Test PDFExporter gracefully handles missing ReportLab."""
        with patch('wordcloud.utils.export.REPORTLAB_AVAILABLE', False):
            exporter = PDFExporter()
            
            with pytest.raises(RuntimeError, match="ReportLab is required"):
                exporter.export_pdf(Mock(), "test.pdf")
    
    def test_gif_exporter_without_pil(self):
        """Test AnimatedGIFExporter handles missing PIL."""
        with patch('wordcloud.utils.export.PIL_AVAILABLE', False):
            exporter = AnimatedGIFExporter()
            
            with pytest.raises(RuntimeError, match="PIL is required"):
                exporter.export_animated_gif([Mock()], "test.gif")

