"""
Template and preset system for wordcloud generation.

This module provides predefined templates and presets for common
wordcloud use cases, allowing users to quickly create wordclouds
with optimized settings.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml

from .logging_config import get_logger

logger = get_logger(__name__)

# Predefined presets
PRESETS = {
    'minimal': {
        'width': 800,
        'height': 400,
        'max_words': 50,
        'min_font_size': 20,
        'max_font_size': 60,
        'background_color': 'white',
        'place_strategy': 'random',
        'prefer_horizontal': 1.0,
        'margin': 2,
    },
    'dense': {
        'width': 1200,
        'height': 800,
        'max_words': 300,
        'min_font_size': 10,
        'max_font_size': 80,
        'background_color': 'white',
        'place_strategy': 'archimedian',
        'prefer_horizontal': 0.8,
        'margin': 1,
    },
    'presentation': {
        'width': 1920,
        'height': 1080,
        'max_words': 100,
        'min_font_size': 24,
        'max_font_size': 120,
        'background_color': '#1a1a1a',
        'place_strategy': 'rectangular',
        'prefer_horizontal': 0.9,
        'margin': 5,
        'black_white': False,
    },
    'social_media': {
        'width': 1080,
        'height': 1080,
        'max_words': 75,
        'min_font_size': 18,
        'max_font_size': 90,
        'background_color': 'white',
        'place_strategy': 'circular',
        'prefer_horizontal': 0.7,
        'margin': 3,
    },
    'print': {
        'width': 2400,
        'height': 3000,
        'max_words': 200,
        'min_font_size': 16,
        'max_font_size': 100,
        'background_color': 'white',
        'place_strategy': 'hierarchical',
        'prefer_horizontal': 0.85,
        'margin': 4,
    },
    'web': {
        'width': 1200,
        'height': 600,
        'max_words': 150,
        'min_font_size': 14,
        'max_font_size': 70,
        'background_color': '#ffffff',
        'place_strategy': 'force_directed',
        'prefer_horizontal': 0.75,
        'margin': 2,
    },
    'artistic': {
        'width': 1600,
        'height': 1200,
        'max_words': 250,
        'min_font_size': 12,
        'max_font_size': 100,
        'background_color': '#000000',
        'place_strategy': 'pytag',
        'prefer_horizontal': 0.5,
        'margin': 3,
        'rotation_angles': (0, 45, 90, -45, -90),
    },
    'compact': {
        'width': 600,
        'height': 400,
        'max_words': 100,
        'min_font_size': 12,
        'max_font_size': 50,
        'background_color': 'white',
        'place_strategy': 'grid',
        'prefer_horizontal': 1.0,
        'margin': 1,
    },
}


class PresetManager:
    """
    Manager for wordcloud presets and templates.
    """
    def __init__(self):
        self.presets = PRESETS.copy()
        self.custom_presets: Dict[str, Dict[str, Any]] = {}
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a preset by name.
        
        Args:
            name: Preset name
            
        Returns:
            Preset dictionary or None if not found
        """
        preset = self.presets.get(name) or self.custom_presets.get(name)
        if preset:
            logger.debug(f"Loaded preset: {name}")
        return preset
    
    def list_presets(self) -> list[str]:
        """
        List all available preset names.
        
        Returns:
            List of preset names
        """
        all_presets = list(self.presets.keys()) + list(self.custom_presets.keys())
        return sorted(all_presets)
    
    def add_custom_preset(self, name: str, preset: Dict[str, Any]) -> None:
        """
        Add a custom preset.
        
        Args:
            name: Preset name
            preset: Preset dictionary
        """
        self.custom_presets[name] = preset
        logger.info(f"Added custom preset: {name}")
    
    def remove_custom_preset(self, name: str) -> bool:
        """
        Remove a custom preset.
        
        Args:
            name: Preset name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.custom_presets:
            del self.custom_presets[name]
            logger.info(f"Removed custom preset: {name}")
            return True
        return False
    
    def save_preset_to_file(self, name: str, file_path: Path, format: str = 'json') -> None:
        """
        Save a preset to a file.
        
        Args:
            name: Preset name
            file_path: Output file path
            format: File format ('json' or 'yaml')
        """
        preset = self.get_preset(name)
        if not preset:
            raise ValueError(f"Preset '{name}' not found")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(preset, f, indent=2)
        elif format == 'yaml':
            try:
                with open(file_path, 'w') as f:
                    yaml.dump(preset, f, default_flow_style=False)
            except ImportError:
                raise RuntimeError("PyYAML required for YAML export. Install with 'pip install pyyaml'")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved preset '{name}' to {file_path}")
    
    def load_preset_from_file(self, file_path: Path, name: Optional[str] = None) -> str:
        """
        Load a preset from a file.
        
        Args:
            file_path: Input file path
            name: Optional preset name (uses filename if None)
            
        Returns:
            Preset name
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Preset file not found: {file_path}")
        
        if name is None:
            name = file_path.stem
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                preset = json.load(f)
        elif file_path.suffix.lower() in ('.yaml', '.yml'):
            try:
                with open(file_path, 'r') as f:
                    preset = yaml.safe_load(f)
            except ImportError:
                raise RuntimeError("PyYAML required for YAML import. Install with 'pip install pyyaml'")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.add_custom_preset(name, preset)
        return name
    
    def create_preset_from_wordcloud(self, wordcloud_instance, name: str) -> None:
        """
        Create a preset from an existing wordcloud instance.
        
        Args:
            wordcloud_instance: Wordcloud instance
            name: Preset name
        """
        preset = {
            'width': wordcloud_instance.width,
            'height': wordcloud_instance.height,
            'max_words': wordcloud_instance.max_words,
            'min_font_size': wordcloud_instance.min_font_size,
            'max_font_size': wordcloud_instance.max_font_size,
            'background_color': wordcloud_instance.background_color,
            'place_strategy': wordcloud_instance.place_strategy,
            'prefer_horizontal': wordcloud_instance.prefer_horizontal,
            'margin': wordcloud_instance.margin,
            'rotation_angles': wordcloud_instance.rotation_angles,
        }
        
        # Add optional parameters if they're set
        if hasattr(wordcloud_instance, 'language') and wordcloud_instance.language:
            preset['language'] = wordcloud_instance.language
        
        if hasattr(wordcloud_instance, 'font_distribution'):
            preset['font_distribution'] = wordcloud_instance.font_distribution
        
        if hasattr(wordcloud_instance, 'enable_stemming'):
            preset['enable_stemming'] = wordcloud_instance.enable_stemming
        
        if hasattr(wordcloud_instance, 'enable_lemmatization'):
            preset['enable_lemmatization'] = wordcloud_instance.enable_lemmatization
        
        self.add_custom_preset(name, preset)


# Global preset manager instance
_preset_manager = PresetManager()


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """Get a preset by name."""
    return _preset_manager.get_preset(name)


def list_presets() -> list[str]:
    """List all available presets."""
    return _preset_manager.list_presets()


def create_wordcloud_from_preset(preset_name: str, **overrides):
    """
    Create a Wordcloud instance from a preset.
    
    Args:
        preset_name: Name of the preset
        **overrides: Parameters to override in the preset
        
    Returns:
        Wordcloud instance
    """
    from ..wordcloud import Wordcloud
    
    preset = get_preset(preset_name)
    if not preset:
        raise ValueError(f"Preset '{preset_name}' not found. Available: {list_presets()}")
    
    # Merge preset with overrides
    params = {**preset, **overrides}
    
    return Wordcloud(**params)

