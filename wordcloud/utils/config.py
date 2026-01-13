#!/usr/bin/env python3
"""
Configuration Management for WordCloud

This module provides utilities for managing configuration settings for the
WordCloud library, allowing users to define default settings in a
configuration file and load them at runtime.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .logging_config import get_logger
from .helpers import ensure_parent_dir

logger = get_logger(__name__)

# Try to import yaml, but don't require it
try:
    import yaml # type: ignore
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Default locations for configuration files
DEFAULT_CONFIG_LOCATIONS = [
    Path.cwd() / "wordcloud_config.yaml",
    Path.cwd() / "wordcloud_config.json",
    Path.home() / ".config" / "wordcloud" / "config.yaml",
    Path.home() / ".config" / "wordcloud" / "config.json",
]

DEFAULT_CONFIG = {
    "wordcloud": {
        "width": 800,
        "height": 500,
        "font_path": "fonts/Arial Unicode.ttf",
        "margin": 2,
        "max_words": 200,
        "min_word_length": 3,
        "min_font_size": 14,
        "max_font_size": 80,
        "font_step": 2,
        "background_color": "white",
        "mode": "RGB",
        "black_white": False,
        "place_strategy": "random",
        "rect_only": False,
        "tracing_files": False,
        "mask_threshold": 200,
        "color_theme": None,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 5000,
        "max_job_age": 3600,
        "max_jobs": 100,
    },
    "performance": {
        "enable_tracking": False,
        "tracking_detail": "basic",
    },
    "output": {
        "default_format": "png",
        "results_folder": "Results",
        "dpi": 300,
    },
}


class ConfigManager:
    """
    Configuration management for WordCloud.
    
    This class handles loading, saving, and merging configuration settings
    from various sources including configuration files, environment variables,
    and programmatically set values.
    
    Attributes:
        config (dict): The current configuration settings
        config_file (Path): Path to the loaded configuration file
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to a configuration file to load.
                If None, the default configuration locations will be searched.
        """
        self.config = dict(DEFAULT_CONFIG)
        self.config_file = None
        
        if config_file:
            self.load_config(config_file)
        else:
            self._load_from_default_locations()
            
        self._load_from_environment()
        
    def _load_from_default_locations(self) -> bool:
        """
        Attempt to load configuration from default locations.
        
        Returns:
            True if a configuration was loaded, False otherwise
        """
        for location in DEFAULT_CONFIG_LOCATIONS:
            if location.exists():
                try:
                    self.load_config(location)
                    return True
                except Exception as e:
                    logger.warning(f"Error loading config from {location}: {e}")
        return False
    
    def _load_from_environment(self) -> None:
        """
        Load configuration values from environment variables.
        
        Environment variables should be prefixed with 'WORDCLOUD_' and use
        uppercase names with underscores between sections.
        """
        prefix = "WORDCLOUD_"
        for key in os.environ:
            if key.startswith(prefix):
                parts = key[len(prefix):].lower().split('_')
                
                config_section = self.config
                for part in parts[:-1]:
                    if part not in config_section:
                        config_section[part] = {}
                    config_section = config_section[part]
                
                value = os.environ[key]
                try:
                    config_section[parts[-1]] = json.loads(value)
                except json.JSONDecodeError:
                    config_section[parts[-1]] = value
    
    def load_config(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_file: Path to the configuration file (.yaml or .json)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported
        """
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        logger.info(f"Loading configuration from {path}")
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ValueError("YAML support not available. Install pyyaml.")
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
            
            if config_data:
                self._merge_config(self.config, config_data)
                
            self.config_file = path
            logger.info(f"Configuration loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {e}")
            raise
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge a source configuration into a target configuration.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, output_file: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the current configuration to a file.
        
        Args:
            output_file: Path where to save the configuration.
                
        Returns:
            Path to the saved configuration file
        """
        if not output_file:
            if self.config_file:
                output_file = self.config_file
            else:
                from .helpers import create_folder
                config_dir = create_folder(Path.home() / ".config" / "wordcloud")
                output_file = config_dir / "config.json"
        
        path = ensure_parent_dir(output_file)
        
        try:
            
            if path.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                with open(path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            else:
                path = path.with_suffix('.json') if path.suffix.lower() not in ['.json'] else path
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e}")
            raise IOError(f"Failed to save configuration: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value, with an optional default.
        """
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def get_wordcloud_defaults(self) -> Dict[str, Any]:
        """
        Get default parameters for the Wordcloud class.
        """
        return dict(self.config.get('wordcloud', {}))
    
    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API server configuration.
        """
        return dict(self.config.get('api', {}))
    
    def get_performance_config(self) -> Dict[str, Any]:
        """
        Get performance-related configuration.
        """
        return dict(self.config.get('performance', {}))

    @classmethod
    def get_global_config(cls) -> 'ConfigManager':
        """
        Get or create a global ConfigManager instance.
        """
        if not hasattr(cls, '_instance'):
            cls._instance = ConfigManager()
        return cls._instance


_config_manager = None


def get_config() -> ConfigManager:
    """
    Get the global ConfigManager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

