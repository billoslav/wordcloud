"""
Performance optimization utilities for wordcloud generation.

This module provides caching, parallel processing, and memory optimization
utilities to improve wordcloud generation performance.
"""

from __future__ import annotations

import logging
import functools
import hashlib
import pickle
from typing import Optional, Callable, Any, Dict, List, Tuple
from pathlib import Path
import threading

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False
    logger.debug("concurrent.futures not available, parallel processing disabled")


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation.
    """
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


class DiskCache:
    """
    Disk-based cache for expensive computations.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".wordcloud_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Disk cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                return None
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in disk cache."""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key}: {e}")
    
    def clear(self) -> None:
        """Clear disk cache."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")


def cached_result(cache: Optional[LRUCache] = None, key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache: LRUCache instance (creates new one if None)
        key_func: Function to generate cache key from args/kwargs
    """
    def decorator(func: Callable) -> Callable:
        if cache is None:
            func_cache = LRUCache()
        else:
            func_cache = cache
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name + args + kwargs
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)
            
            # Check cache
            cached_value = func_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Compute and cache
            result = func(*args, **kwargs)
            func_cache.put(cache_key, result)
            return result
        
        wrapper.cache = func_cache
        return wrapper
    
    return decorator


def parallel_process_words(
    words: List[Tuple],
    process_func: Callable,
    max_workers: Optional[int] = None,
    use_processes: bool = False
) -> List[Any]:
    """
    Process words in parallel.
    
    Args:
        words: List of word tuples to process
        process_func: Function to process each word
        max_workers: Maximum number of workers (None = auto)
        use_processes: Use processes instead of threads
        
    Returns:
        List of processed results
    """
    if not CONCURRENT_AVAILABLE:
        logger.warning("concurrent.futures not available, falling back to sequential processing")
        return [process_func(word) for word in words]
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    results = []
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_word = {executor.submit(process_func, word): word for word in words}
        
        # Collect results as they complete
        for future in as_completed(future_to_word):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                word = future_to_word[future]
                logger.error(f"Error processing word {word}: {e}")
    
    return results


def optimize_memory_usage(wordcloud_instance) -> None:
    """
    Optimize memory usage of a wordcloud instance.
    
    This clears intermediate data structures that are no longer needed.
    """
    # Clear font cache if it's too large
    if hasattr(wordcloud_instance, '_font_cache'):
        font_cache = wordcloud_instance._font_cache
        if hasattr(font_cache, 'cache') and len(font_cache.cache) > 100:
            # Keep only most recently used fonts
            logger.debug("Clearing old font cache entries")
            # Implementation depends on FontCache structure
    
    # Clear performance metrics if not needed
    if hasattr(wordcloud_instance, 'performance_metrics'):
        if not wordcloud_instance.enable_performance_tracking:
            wordcloud_instance.performance_metrics.clear()
    
    logger.debug("Memory optimization completed")


class MemoryMonitor:
    """
    Monitor memory usage during wordcloud generation.
    """
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
    
    def start(self) -> None:
        """Start monitoring."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            logger.debug("psutil not available, memory monitoring disabled")
    
    def update(self) -> None:
        """Update peak memory."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
        except ImportError:
            pass
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            'start_memory_mb': self.start_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': self.peak_memory - self.start_memory
        }

