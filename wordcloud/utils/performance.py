"""
Performance utilities for the wordcloud library.

This module provides performance monitoring, profiling, and optimization
tools for the wordcloud generator.
"""

import time
import logging
import sys
import cProfile
import pstats
import io
from typing import Dict, Any, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .logging_config import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Simple context manager for timing code blocks.
    
    Example:
        with Timer() as timer:
            # Code to time
            result = process_data()
        
        print(f"Processing took {timer.elapsed:.2f} seconds")
    """
    
    def __init__(self, name: Optional[str] = None, log_level: str = 'INFO'):
        """
        Initialize a timer.
        
        Args:
            name: Optional name for this timer (for logging)
            log_level: Logging level to use when reporting time
        """
        self.name = name
        self.log_level = log_level
        self.start_time = 0
        self.elapsed = 0
    
    def __enter__(self):
        """Start the timer when entering the context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop the timer when exiting the context."""
        self.elapsed = time.time() - self.start_time
        
        if self.name:
            if self.log_level == 'DEBUG':
                logger.debug(f"{self.name} took {self.elapsed:.4f} seconds")
            elif self.log_level == 'INFO':
                logger.info(f"{self.name} took {self.elapsed:.4f} seconds")


class Profiler:
    """
    Profiler class for analyzing code performance.
    
    Example:
        profiler = Profiler()
        stats = profiler.profile(my_function, arg1, arg2, kwarg1=value1)
        
        # Print top 10 functions by cumulative time
        profiler.print_stats(10)
        
        # Get profiling data as dictionary
        data = profiler.get_stats_dict(20)
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.profiler = cProfile.Profile()
        self.stats = None
    
    def profile(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profile a function call.
        
        Args:
            func: The function to profile
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The return value of the profiled function
        """
        self.profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            self.profiler.disable()
            
        # Process the stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        self.stats = ps
        
        return result
    
    def print_stats(self, limit: int = 20) -> None:
        """
        Print profiling statistics.
        
        Args:
            limit: Maximum number of functions to print
        """
        if self.stats:
            self.stats.print_stats(limit)
        else:
            logger.warning("No profiling data available. Run profile() first.")
    
    def get_stats_dict(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get profiling statistics as a dictionary.
        
        Args:
            limit: Maximum number of functions to include
            
        Returns:
            Dictionary with profiling data
        """
        if not self.stats:
            logger.warning("No profiling data available. Run profile() first.")
            return {}
        
        # Redirect output to a string stream
        s = io.StringIO()
        self.stats.stream = s
        self.stats.print_stats(limit)
        
        # Extract total time
        total_time = self.stats.total_tt
        
        # Process the stats into a dictionary
        result = {
            'total_time': total_time,
            'functions': []
        }
        
        # Extract function stats
        for func, (cc, nc, tt, ct, callers) in self.stats.stats.items():
            if len(result['functions']) >= limit:
                break
                
            # Get file, line, and function name
            if len(func) == 3:
                file_path, line_num, func_name = func
            else:
                file_path, line_num, func_name = "unknown", 0, str(func)
            
            # Add to result
            result['functions'].append({
                'file': file_path,
                'line': line_num,
                'function': func_name,
                'calls': nc,
                'time': tt,
                'time_per_call': tt / nc if nc > 0 else 0,
                'cumulative_time': ct,
                'cumulative_time_pct': (ct / total_time) * 100 if total_time > 0 else 0
            })
        
        return result


def benchmark_functions(funcs: Dict[str, Callable], *args, repeat: int = 3, **kwargs) -> Dict[str, Any]:
    """
    Benchmark multiple functions with the same arguments.
    
    Args:
        funcs: Dictionary mapping function names to functions
        *args: Arguments to pass to each function
        repeat: Number of times to repeat each function
        **kwargs: Keyword arguments to pass to each function
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'fastest': None,
        'slowest': None,
        'function_timings': {}
    }
    
    fastest_time = float('inf')
    slowest_time = 0
    
    for name, func in funcs.items():
        logger.info(f"Benchmarking {name}...")
        
        # Run multiple times and take average
        timings = []
        for i in range(repeat):
            with Timer() as t:
                func(*args, **kwargs)
            timings.append(t.elapsed)
        
        # Calculate statistics
        avg_time = sum(timings) / len(timings)
        results['function_timings'][name] = {
            'avg_time': avg_time,
            'min_time': min(timings),
            'max_time': max(timings),
            'timings': timings
        }
        
        # Update fastest/slowest
        if avg_time < fastest_time:
            fastest_time = avg_time
            results['fastest'] = name
        
        if avg_time > slowest_time:
            slowest_time = avg_time
            results['slowest'] = name
    
    # Calculate relative performance
    for name in results['function_timings']:
        avg_time = results['function_timings'][name]['avg_time']
        fastest_func_time = results['function_timings'][results['fastest']]['avg_time']
        
        if fastest_func_time > 0:
            relative = avg_time / fastest_func_time
        else:
            relative = float('inf')
            
        results['function_timings'][name]['relative'] = relative
    
    return results


def optimize_parameters(
    func: Callable,
    param_ranges: Dict[str, List[Any]],
    evaluation_func: Callable[[Any], float],
    *args,
    max_tests: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Find optimal parameters for a function by testing combinations.
    
    Args:
        func: Function to optimize
        param_ranges: Dictionary mapping parameter names to lists of possible values
        evaluation_func: Function that takes the result of func and returns a score
            (higher is better)
        *args: Additional positional arguments to pass to func
        max_tests: Maximum number of parameter combinations to test
        **kwargs: Additional keyword arguments to pass to func
        
    Returns:
        Dictionary with optimization results
    """
    import itertools
    import random
    
    # Get all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(itertools.product(*[param_ranges[name] for name in param_names]))
    
    # Limit the number of tests
    if len(param_values) > max_tests:
        logger.info(f"Too many parameter combinations ({len(param_values)}). "
                   f"Sampling {max_tests} random combinations.")
        param_values = random.sample(param_values, max_tests)
    
    results = {
        'best_score': float('-inf'),
        'optimal_params': {},
        'all_results': []
    }
    
    # Use ProcessPoolExecutor for parallel testing
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Testing {len(param_values)} parameter combinations using {num_cores} cores...")
    
    def test_params(params_tuple):
        """Test a single parameter combination."""
        # Convert tuple of values to dictionary
        params = {name: value for name, value in zip(param_names, params_tuple)}
        
        # Create complete kwargs dict
        test_kwargs = kwargs.copy()
        test_kwargs.update(params)
        
        # Time and run the function
        start_time = time.time()
        result = func(*args, **test_kwargs)
        elapsed = time.time() - start_time
        
        # Evaluate the result
        score = evaluation_func(result)
        
        return {
            'params': params,
            'score': score,
            'time': elapsed
        }
    
    # Run tests in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(test_params, params) for params in param_values]
        
        for future in as_completed(futures):
            try:
                test_result = future.result()
                results['all_results'].append(test_result)
                
                # Update best result if better
                if test_result['score'] > results['best_score']:
                    results['best_score'] = test_result['score']
                    results['optimal_params'] = test_result['params']
                    
                    logger.info(f"New best score: {test_result['score']}")
                    logger.info(f"Parameters: {test_result['params']}")
            except Exception as e:
                logger.error(f"Error during parameter optimization: {e}")
    
    # Sort all results by score
    results['all_results'].sort(key=lambda x: x['score'], reverse=True)
    
    return results


def profile_memory_usage(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile memory usage of a function.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with memory usage data
    """
    try:
        import tracemalloc
        import gc
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Start tracking memory allocations
        tracemalloc.start()
        
        # Get current memory usage
        start_snapshot = tracemalloc.take_snapshot()
        start_stats = start_snapshot.statistics('lineno')
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Get memory usage after function execution
        peak_usage = tracemalloc.get_traced_memory()[1]
        end_snapshot = tracemalloc.take_snapshot()
        
        # Stop tracking
        tracemalloc.stop()
        
        # Compare snapshots
        end_stats = end_snapshot.statistics('lineno')
        
        # Process statistics
        memory_stats = []
        for stat in end_stats[:20]:  # Top 20 memory consumers
            memory_stats.append({
                'file': stat.traceback[0].filename,
                'line': stat.traceback[0].lineno,
                'size': stat.size,
                'count': stat.count
            })
        
        return {
            'peak_memory_usage': peak_usage,
            'peak_memory_mb': peak_usage / (1024 * 1024),
            'top_consumers': memory_stats,
            'result': result
        }
        
    except ImportError:
        logger.warning("tracemalloc module not available. Memory profiling disabled.")
        
        # Run function without memory profiling
        result = func(*args, **kwargs)
        return {
            'peak_memory_usage': None,
            'peak_memory_mb': None,
            'top_consumers': [],
            'result': result
        }


class PerformanceTracker:
    """
    Comprehensive performance tracker that combines timing, profiling, and memory tracking.
    
    This class provides a unified interface for tracking performance metrics including
    execution time, function call counts, and memory usage. It can be used as a context
    manager for easy integration.
    
    Example:
        tracker = PerformanceTracker("text_processing", detail_level="detailed")
        with tracker:
            process_text(text)
        
        metrics = tracker.get_summary()
        print(f"Total time: {metrics['total_time']:.2f}s")
    """
    
    def __init__(self, name: Optional[str] = None, detail_level: str = "basic"):
        """
        Initialize a performance tracker.
        
        Args:
            name: Optional name for this tracker (for logging)
            detail_level: Level of detail ("basic" or "detailed")
                - "basic": Only timing information
                - "detailed": Timing + memory + function call counts
        """
        self.name = name or "operation"
        self.detail_level = detail_level
        self.timer = Timer(name=name)
        self.profiler = None
        self.memory_start = None
        self.memory_peak = None
        self.memory_stats = None
        self.is_active = False
        self._profiler_enabled = False
        self._tracemalloc_started = False
        
        if detail_level == "detailed":
            self.profiler = Profiler()
            try:
                import tracemalloc
                self._tracemalloc_available = True
            except ImportError:
                self._tracemalloc_available = False
                logger.warning("tracemalloc not available. Memory tracking will be disabled.")
        else:
            self._tracemalloc_available = False
    
    def __enter__(self):
        """Start tracking when entering the context."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Stop tracking when exiting the context."""
        self.stop()
    
    def start(self) -> None:
        """Start performance tracking."""
        self.is_active = True
        self._profiler_enabled = False
        self._tracemalloc_started = False
        
        # Start timer
        self.timer.__enter__()
        
        # Start profiler if detailed tracking
        if self.profiler:
            # cProfile uses sys.setprofile; nested profilers will raise.
            # If another profiler is already active, skip enabling for this tracker.
            if sys.getprofile() is None:
                try:
                    self.profiler.profiler.enable()
                    self._profiler_enabled = True
                except ValueError:
                    # Some environments install a global profiling hook that still
                    # triggers cProfile's guard even if sys.getprofile() is None.
                    logger.debug("Another profiling tool is already active; skipping profiler enable.")
            else:
                logger.debug("Another profiler is already active; skipping nested profiler enable.")
        
        # Start memory tracking if available
        if self._tracemalloc_available:
            try:
                import tracemalloc
                import gc
                gc.collect()
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                    self._tracemalloc_started = True
                self.memory_start = tracemalloc.take_snapshot()
            except Exception as e:
                logger.warning(f"Failed to start memory tracking: {e}")
                self._tracemalloc_available = False
    
    def stop(self) -> None:
        """Stop performance tracking and collect metrics."""
        if not self.is_active:
            return
        
        # Stop timer
        self.timer.__exit__(None, None, None)
        
        # Stop profiler if detailed tracking
        if self.profiler and self._profiler_enabled:
            self.profiler.profiler.disable()
            # Process stats
            s = io.StringIO()
            ps = pstats.Stats(self.profiler.profiler, stream=s).sort_stats('cumulative')
            self.profiler.stats = ps
        
        # Stop memory tracking
        if self._tracemalloc_available:
            try:
                import tracemalloc
                self.memory_peak = tracemalloc.get_traced_memory()[1]
                end_snapshot = tracemalloc.take_snapshot()
                if self._tracemalloc_started:
                    tracemalloc.stop()
                
                # Process memory statistics
                end_stats = end_snapshot.statistics('lineno')
                self.memory_stats = []
                for stat in end_stats[:20]:  # Top 20 memory consumers
                    self.memory_stats.append({
                        'file': stat.traceback[0].filename,
                        'line': stat.traceback[0].lineno,
                        'size': stat.size,
                        'count': stat.count
                    })
            except Exception as e:
                logger.warning(f"Failed to stop memory tracking: {e}")
        
        self.is_active = False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all performance metrics.
        
        Returns:
            Dictionary containing performance metrics:
            - total_time: Total execution time in seconds
            - memory_peak_mb: Peak memory usage in MB (if detailed tracking)
            - memory_stats: Top memory consumers (if detailed tracking)
            - function_stats: Function call statistics (if detailed tracking)
        """
        summary = {
            'name': self.name,
            'total_time': self.timer.elapsed,
            'detail_level': self.detail_level
        }
        
        if self.detail_level == "detailed":
            # Add memory information
            if self.memory_peak is not None:
                summary['memory_peak_bytes'] = self.memory_peak
                summary['memory_peak_mb'] = self.memory_peak / (1024 * 1024)
            
            if self.memory_stats:
                summary['memory_stats'] = self.memory_stats
            
            # Add function call statistics
            if self.profiler and self.profiler.stats:
                summary['function_stats'] = self.profiler.get_stats_dict(limit=20)
        
        return summary
    
    def log_summary(self, log_level: str = 'INFO') -> None:
        """
        Log a summary of performance metrics.
        
        Args:
            log_level: Logging level to use ('INFO', 'DEBUG', etc.)
        """
        summary = self.get_summary()
        
        log_msg = f"Performance summary for '{summary['name']}': "
        log_msg += f"Total time: {summary['total_time']:.4f}s"
        
        if self.detail_level == "detailed":
            if 'memory_peak_mb' in summary:
                log_msg += f", Peak memory: {summary['memory_peak_mb']:.2f} MB"
            
            if 'function_stats' in summary and summary['function_stats']:
                total_calls = sum(
                    func['calls'] 
                    for func in summary['function_stats'].get('functions', [])
                )
                log_msg += f", Total function calls: {total_calls}"
        
        if log_level == 'DEBUG':
            logger.debug(log_msg)
        elif log_level == 'INFO':
            logger.info(log_msg)
        elif log_level == 'WARNING':
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

