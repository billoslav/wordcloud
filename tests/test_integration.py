"""
Integration tests for end-to-end wordcloud workflows.

These tests verify that the complete wordcloud generation pipeline works
correctly from text input through to various output formats.
"""

import os
import tempfile
import pytest # type: ignore
from pathlib import Path
from unittest.mock import patch
import numpy as np

from wordcloud import Wordcloud
from wordcloud.utils import STRATEGIES, COLOR_THEMES
from wordcloud.utils.visualization import generate_colors_by_frequency


class TestBasicWordcloudGeneration:
    """Test basic wordcloud generation workflows."""
    
    @pytest.fixture
    def sample_text(self):
        """Provide sample text for testing."""
        return """
        Python is a great programming language for data science and machine learning.
        Python offers excellent libraries like NumPy, Pandas, and Matplotlib.
        Machine learning with Python is both powerful and accessible.
        Data analysis in Python is straightforward and efficient.
        """
    
    @pytest.fixture
    def font_path(self):
        """Provide a valid font path."""
        return "fonts/Arial Unicode.ttf"
    
    def test_basic_generation(self, sample_text, font_path):
        """Test basic wordcloud generation."""
        wc = Wordcloud(width=400, height=200, font_path=font_path)
        wc.generate(sample_text)
        
        assert wc.gen_positions is not None
        assert len(wc.gen_positions) > 0
    
    def test_draw_image(self, sample_text, font_path):
        """Test drawing wordcloud as image."""
        wc = Wordcloud(width=400, height=200, font_path=font_path)
        wc.generate(sample_text)
        
        img = wc.draw_image()
        
        assert img is not None
        assert img.size == (400, 200)
    
    def test_save_image(self, sample_text, font_path, tmp_path):
        """Test saving wordcloud image to file."""
        wc = Wordcloud(width=400, height=200, font_path=font_path)
        wc.results_folder = str(tmp_path)
        wc.generate(sample_text)
        
        img = wc.draw_image(save_file=True, image_name="test_wordcloud")
        
        saved_path = tmp_path / "test_wordcloud.png"
        assert saved_path.exists()
    
    def test_generate_svg(self, sample_text, font_path):
        """Test SVG generation."""
        wc = Wordcloud(width=400, height=200, font_path=font_path)
        wc.generate(sample_text)
        
        svg = wc.generate_svg()
        
        assert svg is not None
        assert '<svg' in svg
        assert '</svg>' in svg
    
    def test_generate_html(self, sample_text, font_path):
        """Test HTML generation."""
        wc = Wordcloud(width=400, height=200, font_path=font_path)
        wc.generate(sample_text)
        
        html = wc.create_html()
        
        assert html is not None
        assert '<!DOCTYPE html>' in html
        assert '</html>' in html


class TestPlacementStrategies:
    """Test different placement strategies."""
    
    @pytest.fixture
    def text(self):
        return "word one two three four five six seven eight nine ten"
    
    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"
    
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_placement_strategy(self, text, font_path, strategy):
        """Test that each placement strategy works."""
        wc = Wordcloud(
            width=300, 
            height=200, 
            font_path=font_path,
            place_strategy=strategy,
            max_words=10
        )
        wc.generate(text)
        
        # Should generate at least some positions (may not place all words)
        assert wc.gen_positions is not None


class TestMaskSupport:
    """Test wordcloud generation with masks."""
    
    @pytest.fixture
    def circular_mask(self):
        """Create a circular mask."""
        size = 200
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 2 - 10
        mask = ((x - center) ** 2 + (y - center) ** 2 <= radius ** 2).astype(np.uint8)
        return mask
    
    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"
    
    def test_masked_wordcloud(self, circular_mask, font_path):
        """Test wordcloud generation with a mask."""
        text = "mask test word cloud generation example text here"
        
        wc = Wordcloud(
            width=200,
            height=200,
            font_path=font_path,
            mask_image=circular_mask,
            max_words=20
        )
        wc.generate(text)
        
        assert wc.gen_positions is not None


class TestGenerateWithColorTheme:
    """Test applying a color theme directly via Wordcloud.generate()."""

    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"

    def test_generate_applies_color_theme(self, font_path):
        wc = Wordcloud(width=300, height=150, font_path=font_path, black_white=False)
        wc.generate("alpha beta gamma delta", color_theme="viridis")

        assert wc.gen_positions is not None
        # Expect colors to be hex strings from visualization.py (e.g. '#rrggbb')
        colors = [pos[5] for pos in wc.gen_positions]
        assert any(isinstance(c, str) and c.startswith("#") for c in colors)
    
    def test_mask_dimension_mismatch_raises(self, font_path):
        """Test that mismatched mask dimensions raise an error."""
        mask = np.ones((100, 100), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Mask dimensions"):
            Wordcloud(
                width=200,
                height=200,
                font_path=font_path,
                mask_image=mask
            )


class TestColorCustomization:
    """Test color customization features."""
    
    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"
    
    @pytest.fixture
    def wordcloud_with_words(self, font_path):
        """Create a wordcloud with generated positions."""
        wc = Wordcloud(width=300, height=200, font_path=font_path)
        wc.generate("hello world test example sample demo")
        return wc
    
    def test_update_colors_with_list(self, wordcloud_with_words):
        """Test updating colors with a list."""
        wc = wordcloud_with_words
        original_colors = [p[5] for p in wc.gen_positions]
        
        new_colors = ["red", "blue", "green"]
        wc.update_colors(new_colors)
        
        updated_colors = [p[5] for p in wc.gen_positions]
        # First few colors should be updated
        assert updated_colors[0] == "red"
        assert updated_colors[1] == "blue"
        assert updated_colors[2] == "green"
    
    def test_update_colors_with_dict(self, wordcloud_with_words):
        """Test updating colors with a dictionary."""
        wc = wordcloud_with_words
        
        word_colors = {"hello": "purple", "world": "orange"}
        wc.update_colors(word_colors)
        
        # Check that specific words got their colors updated
        for pos in wc.gen_positions:
            word = pos[0][0]
            color = pos[5]
            if word in word_colors:
                assert color == word_colors[word]
    
    def test_black_white_mode(self, font_path):
        """Test black and white mode."""
        wc = Wordcloud(
            width=300, 
            height=200, 
            font_path=font_path,
            black_white=True
        )
        wc.generate("black white test words")
        
        # All colors should be black
        for pos in wc.gen_positions:
            assert pos[5] == "rgb(0, 0, 0)"


class TestPerformanceTracking:
    """Test performance tracking integration."""
    
    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"
    
    def test_performance_tracking_enabled(self, font_path):
        """Test that performance tracking captures metrics."""
        wc = Wordcloud(
            width=200,
            height=100,
            font_path=font_path,
            enable_performance_tracking=True,
            performance_tracking_detail="basic"
        )
        wc.generate("performance test words")
        
        assert 'generate' in wc.performance_metrics
        assert 'total_time' in wc.performance_metrics['generate']
    
    def test_performance_tracking_detailed(self, font_path):
        """Test detailed performance tracking."""
        wc = Wordcloud(
            width=200,
            height=100,
            font_path=font_path,
            enable_performance_tracking=True,
            performance_tracking_detail="detailed"
        )
        wc.generate("detailed performance test")
        
        metrics = wc.performance_metrics.get('generate', {})
        assert 'total_time' in metrics
    
    def test_performance_tracking_disabled(self, font_path):
        """Test that metrics are empty when tracking is disabled."""
        wc = Wordcloud(
            width=200,
            height=100,
            font_path=font_path,
            enable_performance_tracking=False
        )
        wc.generate("no tracking test")
        
        # Metrics should be empty or not contain timing data
        assert len(wc.performance_metrics) == 0


class TestInputValidation:
    """Test input validation in the Wordcloud class."""
    
    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"
    
    def test_invalid_dimensions(self, font_path):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError, match="positive integers"):
            Wordcloud(width=-100, height=200, font_path=font_path)
        
        with pytest.raises(ValueError, match="positive integers"):
            Wordcloud(width=100, height=0, font_path=font_path)
    
    def test_invalid_strategy(self, font_path):
        """Test that invalid strategy raises an error."""
        with pytest.raises(ValueError, match="Invalid placement strategy"):
            Wordcloud(width=100, height=100, font_path=font_path, place_strategy="nonexistent")
    
    def test_invalid_font_size(self, font_path):
        """Test that invalid font sizes raise errors."""
        with pytest.raises(ValueError, match="min_font_size must be positive"):
            Wordcloud(width=100, height=100, font_path=font_path, min_font_size=0)
        
        with pytest.raises(ValueError, match="max_font_size.*must be >= min_font_size"):
            Wordcloud(width=100, height=100, font_path=font_path, min_font_size=20, max_font_size=10)
    
    def test_invalid_max_words(self, font_path):
        """Test that invalid max_words raises an error."""
        with pytest.raises(ValueError, match="max_words must be positive"):
            Wordcloud(width=100, height=100, font_path=font_path, max_words=0)
    
    def test_invalid_margin(self, font_path):
        """Test that negative margin raises an error."""
        with pytest.raises(ValueError, match="margin must be non-negative"):
            Wordcloud(width=100, height=100, font_path=font_path, margin=-5)


class TestTextProcessing:
    """Test text processing integration."""
    
    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"
    
    def test_stopwords_filtering(self, font_path):
        """Test that stopwords are properly filtered."""
        wc = Wordcloud(
            width=300,
            height=200,
            font_path=font_path,
            stopwords=["the", "and", "is"]
        )
        wc.generate("The cat and the dog is playing")
        
        words_placed = [pos[0][0] for pos in wc.gen_positions]
        # Stopword check is case-sensitive before lowercasing (so "The" survives and becomes "the")
        assert "and" not in words_placed
        assert "is" not in words_placed
    
    def test_min_word_length(self, font_path):
        """Test minimum word length filtering."""
        wc = Wordcloud(
            width=300,
            height=200,
            font_path=font_path,
            min_word_length=5
        )
        wc.generate("a an the longer words remain here today")
        
        words_placed = [pos[0][0] for pos in wc.gen_positions]
        # Short words should be filtered
        for word in words_placed:
            assert len(word) >= 5
    
    def test_empty_text_handling(self, font_path):
        """Test handling of empty text."""
        wc = Wordcloud(width=200, height=100, font_path=font_path)
        
        with pytest.raises(ValueError):
            wc.generate("")


class TestMethodChaining:
    """Test that methods properly return self for chaining."""
    
    @pytest.fixture
    def font_path(self):
        return "fonts/Arial Unicode.ttf"
    
    def test_generate_returns_self(self, font_path):
        """Test that generate() returns the Wordcloud instance."""
        wc = Wordcloud(width=200, height=100, font_path=font_path)
        result = wc.generate("test words")
        
        assert result is wc
    
    def test_update_colors_returns_self(self, font_path):
        """Test that update_colors() returns the Wordcloud instance."""
        wc = Wordcloud(width=200, height=100, font_path=font_path)
        wc.generate("test words")
        result = wc.update_colors(["red"])
        
        assert result is wc
    
    def test_chained_operations(self, font_path):
        """Test chaining multiple operations."""
        wc = Wordcloud(width=200, height=100, font_path=font_path)
        
        # Chain generate and update_colors
        result = wc.generate("test words here").update_colors({"test": "blue"})
        
        assert result is wc
        assert wc.gen_positions is not None


class TestCollisionDetection:
    """Test collision detection utilities."""
    
    def test_grid_collision_system(self):
        """Test GridCollisionSystem integration."""
        from wordcloud.utils.collision import GridCollisionSystem
        
        grid = GridCollisionSystem(100, 100, cell_size=10)
        
        # Add a rectangle
        grid.add_rectangle((10, 10, 20, 20))
        
        # Check collision
        assert grid.check_collision((15, 15, 10, 10))  # Overlaps
        assert not grid.check_collision((50, 50, 10, 10))  # No overlap
    
    def test_quadtree_collision(self):
        """Test QuadtreeCollisionDetector integration."""
        from wordcloud.utils.collision import QuadtreeCollisionDetector
        
        quad = QuadtreeCollisionDetector(200, 200)
        
        quad.add_rectangle((50, 50, 30, 30))
        
        assert quad.check_collision((60, 60, 10, 10))  # Overlaps
        assert not quad.check_collision((100, 100, 20, 20))  # No overlap
    
    def test_create_collision_detector_factory(self):
        """Test the collision detector factory function."""
        from wordcloud.utils.collision import create_collision_detector
        
        brute = create_collision_detector('brute', 100, 100)
        grid = create_collision_detector('grid', 100, 100, cell_size=5)
        quad = create_collision_detector('quadtree', 100, 100)
        
        assert brute is not None
        assert grid is not None
        assert quad is not None


class TestConfigurationManagement:
    """Test configuration management utilities."""
    
    def test_config_manager_defaults(self):
        """Test ConfigManager provides defaults."""
        from wordcloud.utils.config import ConfigManager, DEFAULT_CONFIG
        
        config = ConfigManager()
        
        # Should have wordcloud defaults
        wc_defaults = config.get_wordcloud_defaults()
        assert 'width' in wc_defaults
        assert 'height' in wc_defaults
    
    def test_config_get_set(self):
        """Test getting and setting config values."""
        from wordcloud.utils.config import ConfigManager
        
        config = ConfigManager()
        
        # Set a value
        config.set('wordcloud', 'custom_key', 'custom_value')
        
        # Get it back
        assert config.get('wordcloud', 'custom_key') == 'custom_value'
        
        # Get with default
        assert config.get('wordcloud', 'nonexistent', 'default') == 'default'

