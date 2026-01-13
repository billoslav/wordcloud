"""
Logging configuration utilities for the wordcloud library.

This module provides centralized logging setup with support for console and file
handlers, configurable log levels, and log rotation.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = True,
    log_format: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure logging for the wordcloud library.
    
    Sets up logging with both console and file handlers. The file handler uses
    rotation to prevent log files from growing too large.
    
    Args:
        log_level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Path to log file. If None, uses 'wordcloud.log' in current directory
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        log_format: Custom log format string. If None, uses default format
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
    
    Example:
        >>> setup_logging(log_level=logging.DEBUG, log_file="my_app.log")
    """
    # Default log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        if log_file is None:
            log_file = Path.cwd() / "wordcloud.log"
        
        # Ensure log directory exists (lazy import to avoid circular dependency)
        from .helpers import ensure_parent_dir
        log_file = ensure_parent_dir(log_file)
        
        # Use RotatingFileHandler for log rotation
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    This is a convenience function that ensures the logger is properly configured.
    If logging hasn't been set up yet, it will use basic configuration.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a log message")
    """
    logger = logging.getLogger(name)
    
    # If no handlers are configured, set up basic logging
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    return logger

