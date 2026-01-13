"""
Path utility functions for normalizing and handling file paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..logging_config import get_logger

logger = get_logger(__name__)


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Convert a path to a Path object consistently.
    
    Args:
        path: Path as string or Path object
        
    Returns:
        Path object
        
    Example:
        >>> normalize_path("some/path")
        Path('some/path')
        >>> normalize_path(Path("some/path"))
        Path('some/path')
    """
    return Path(path) if isinstance(path, str) else path


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """
    Ensure the parent directory of a path exists, creating it if necessary.
    
    Creates the parent directory structure if it doesn't exist.
    This is useful when saving files to ensure the directory structure exists.
    
    Args:
        path: File path (string or Path) whose parent directory should be created
        
    Returns:
        Normalized Path object
        
    Example:
        >>> ensure_parent_dir("output/images/wordcloud.png")
        Path('output/images/wordcloud.png')
        # Creates 'output/images/' directory if it doesn't exist
    """
    normalized_path = normalize_path(path)
    
    if normalized_path.parent:
        try:
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Parent directory '{normalized_path.parent}' created/already exists.")
        except OSError as e:
            logger.error(f"Error creating parent directory '{normalized_path.parent}': {e}")
    
    return normalized_path

