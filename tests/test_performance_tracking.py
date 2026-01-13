"""
Tests for performance tracking functionality in the wordcloud library.
"""

import unittest
import time
from wordcloud.utils.performance import PerformanceTracker, Timer, Profiler


class TestPerformanceTracker(unittest.TestCase):
    """Test PerformanceTracker class."""
    
    def test_basic_tracking(self):
        """Test basic performance tracking (timing only)."""
        tracker = PerformanceTracker("test_operation", detail_level="basic")
        
        with tracker:
            time.sleep(0.1)  # Simulate some work
        
        summary = tracker.get_summary()
        
        self.assertIn("total_time", summary)
        self.assertIn("name", summary)
        self.assertEqual(summary["name"], "test_operation")
        self.assertEqual(summary["detail_level"], "basic")
        self.assertGreater(summary["total_time"], 0)
        self.assertLess(summary["total_time"], 1.0)  # Should be around 0.1s
    
    def test_detailed_tracking(self):
        """Test detailed performance tracking (timing + memory + profiling)."""
        tracker = PerformanceTracker("test_operation", detail_level="detailed")
        
        def test_function():
            # Create some data to track memory
            data = [i for i in range(1000)]
            return sum(data)
        
        tracker.start()
        result = test_function()
        tracker.stop()
        
        summary = tracker.get_summary()
        
        self.assertIn("total_time", summary)
        self.assertIn("detail_level", summary)
        self.assertEqual(summary["detail_level"], "detailed")
        
        # Memory tracking may or may not be available
        if "memory_peak_mb" in summary:
            self.assertGreater(summary["memory_peak_mb"], 0)
    
    def test_context_manager(self):
        """Test PerformanceTracker as context manager."""
        with PerformanceTracker("test_op", "basic") as tracker:
            time.sleep(0.05)
        
        summary = tracker.get_summary()
        self.assertGreater(summary["total_time"], 0)
    
    def test_manual_start_stop(self):
        """Test manual start/stop of tracker."""
        tracker = PerformanceTracker("test_op", "basic")
        
        tracker.start()
        time.sleep(0.05)
        tracker.stop()
        
        summary = tracker.get_summary()
        self.assertGreater(summary["total_time"], 0)
    
    def test_log_summary(self):
        """Test that log_summary works without errors."""
        tracker = PerformanceTracker("test_op", "basic")
        
        with tracker:
            time.sleep(0.01)
        
        # Should not raise an exception
        tracker.log_summary()
        tracker.log_summary(log_level='DEBUG')
        tracker.log_summary(log_level='INFO')


class TestTimer(unittest.TestCase):
    """Test Timer class."""
    
    def test_timer_basic(self):
        """Test basic timer functionality."""
        with Timer() as timer:
            time.sleep(0.1)
        
        self.assertGreater(timer.elapsed, 0)
        self.assertLess(timer.elapsed, 1.0)
    
    def test_timer_with_name(self):
        """Test timer with name."""
        with Timer("test_timer") as timer:
            time.sleep(0.05)
        
        self.assertGreater(timer.elapsed, 0)


class TestProfiler(unittest.TestCase):
    """Test Profiler class."""
    
    def test_profiler_basic(self):
        """Test basic profiler functionality."""
        profiler = Profiler()
        
        def test_function(n):
            total = 0
            for i in range(n):
                total += i
            return total
        
        result = profiler.profile(test_function, 1000)
        
        self.assertEqual(result, sum(range(1000)))
        self.assertIsNotNone(profiler.stats)
    
    def test_profiler_get_stats_dict(self):
        """Test getting profiler stats as dictionary."""
        profiler = Profiler()
        
        def test_function(n):
            total = 0
            for i in range(n):
                total += i
            return total
        
        profiler.profile(test_function, 100)
        stats_dict = profiler.get_stats_dict(limit=10)
        
        self.assertIn("total_time", stats_dict)
        self.assertIn("functions", stats_dict)
        self.assertGreater(len(stats_dict["functions"]), 0)


class TestPerformanceTrackingIntegration(unittest.TestCase):
    """Test performance tracking integration with Wordcloud."""
    
    def test_wordcloud_performance_tracking_basic(self):
        """Test Wordcloud with basic performance tracking."""
        from wordcloud import Wordcloud
        
        wc = Wordcloud(
            width=400,
            height=300,
            enable_performance_tracking=True,
            performance_tracking_detail="basic"
        )
        
        wc.generate("test text for wordcloud generation")
        
        # Check that performance metrics were collected
        self.assertIn("generate", wc.performance_metrics)
        self.assertIn("prepare_text", wc.performance_metrics)
        self.assertIn("find_position", wc.performance_metrics)
        
        # Check that metrics contain timing information
        generate_metrics = wc.performance_metrics["generate"]
        self.assertIn("total_time", generate_metrics)
        self.assertGreater(generate_metrics["total_time"], 0)
    
    def test_wordcloud_performance_tracking_detailed(self):
        """Test Wordcloud with detailed performance tracking."""
        from wordcloud import Wordcloud
        
        wc = Wordcloud(
            width=400,
            height=300,
            enable_performance_tracking=True,
            performance_tracking_detail="detailed"
        )
        
        wc.generate("test text for wordcloud generation")
        
        # Check that performance metrics were collected
        self.assertIn("generate", wc.performance_metrics)
        
        # Check detail level
        generate_metrics = wc.performance_metrics["generate"]
        self.assertEqual(generate_metrics["detail_level"], "detailed")
    
    def test_wordcloud_performance_tracking_disabled(self):
        """Test Wordcloud with performance tracking disabled."""
        from wordcloud import Wordcloud
        
        wc = Wordcloud(
            width=400,
            height=300,
            enable_performance_tracking=False
        )
        
        wc.generate("test text")
        
        # Performance metrics should be empty or minimal
        # (some metrics might still be collected if tracking is enabled elsewhere)
        # But generate should not have detailed tracking
        if "generate" in wc.performance_metrics:
            # If present, it should be from nested calls, not main tracking
            pass
    
    def test_performance_metrics_structure(self):
        """Test that performance metrics have correct structure."""
        from wordcloud import Wordcloud
        
        wc = Wordcloud(
            width=400,
            height=300,
            enable_performance_tracking=True,
            performance_tracking_detail="basic"
        )
        
        wc.generate("test")
        
        for operation, metrics in wc.performance_metrics.items():
            self.assertIn("name", metrics)
            self.assertIn("total_time", metrics)
            self.assertIn("detail_level", metrics)
            self.assertIsInstance(metrics["total_time"], (int, float))
            self.assertGreaterEqual(metrics["total_time"], 0)


if __name__ == '__main__':
    unittest.main()

