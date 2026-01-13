"""
File and directory utility functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..logging_config import get_logger
from .path_utils import normalize_path

logger = get_logger(__name__)


def create_folder(path: Union[str, Path], parents: bool = True) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Creates a folder at the specified path, handling potential errors gracefully.
    This is a unified implementation for directory creation used throughout
    the wordcloud library.
    
    Args:
        path: Path to the directory to create (string or Path)
        parents: If True, create parent directories as needed (default: True)
        
    Returns:
        Path object pointing to the created/existing directory
        
    Notes:
        - Uses Path.mkdir with exist_ok=True to avoid race conditions
        - Logs status messages about folder creation or errors
        
    Example:
        >>> create_folder("output/results")
        Path('output/results')
        # Creates 'output/results' directory if it doesn't exist
    """
    normalized_path = normalize_path(path)
    
    try:
        normalized_path.mkdir(parents=parents, exist_ok=True)
        logger.debug(f"Directory '{normalized_path}' created/already exists.")
    except OSError as e:
        logger.error(f"Error creating directory '{normalized_path}': {e}")
        # Re-raise if parents=False to allow callers to handle the error
        if not parents:
            raise
    
    return normalized_path


def create_tracking_structure(base_dir: Union[str, Path], strategy: str) -> Path:
    """
    Create the directory structure for tracking/tracing images.
    
    Sets up the necessary directories for storing tracking images,
    organized by strategy name. Creates both the base directory and
    a subdirectory for the specific strategy.
    
    Args:
        base_dir: Base directory for tracking images (string or Path)
        strategy: Name of the placement strategy
        
    Returns:
        Path object pointing to the strategy subdirectory
        
    Example:
        >>> create_tracking_structure("Tracking", "random")
        Path('Tracking/random')
        # Creates 'Tracking' and 'Tracking/random' directories if needed
    """
    base_path = normalize_path(base_dir)
    
    # Create base directory
    create_folder(base_path, parents=True)
    
    # Create strategy subdirectory
    strategy_dir = base_path / strategy
    create_folder(strategy_dir, parents=True)
    
    return strategy_dir

