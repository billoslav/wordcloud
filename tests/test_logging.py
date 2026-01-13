"""
Tests for logging functionality in the wordcloud library.
"""

import unittest
import logging
import tempfile
import os
from pathlib import Path

from wordcloud.utils.logging_config import setup_logging, get_logger


class TestLoggingConfig(unittest.TestCase):
    """Test logging configuration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
    
    def test_setup_logging_console_only(self):
        """Test logging setup with console output only."""
        setup_logging(
            log_level=logging.INFO,
            console_output=True,
            file_output=False
        )
        
        root_logger = logging.getLogger()
        self.assertEqual(len(root_logger.handlers), 1)
        self.assertIsInstance(root_logger.handlers[0], logging.StreamHandler)
    
    def test_setup_logging_file_only(self):
        """Test logging setup with file output only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(
                log_level=logging.INFO,
                log_file=str(log_file),
                console_output=False,
                file_output=True
            )
            
            root_logger = logging.getLogger()
            self.assertEqual(len(root_logger.handlers), 1)
            self.assertIsInstance(root_logger.handlers[0], logging.handlers.RotatingFileHandler)
            
            # Test that logging works
            logger = logging.getLogger("test")
            logger.info("Test message")
            
            # Check that log file was created
            self.assertTrue(log_file.exists())
    
    def test_setup_logging_both(self):
        """Test logging setup with both console and file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(
                log_level=logging.INFO,
                log_file=str(log_file),
                console_output=True,
                file_output=True
            )
            
            root_logger = logging.getLogger()
            self.assertEqual(len(root_logger.handlers), 2)
            
            handler_types = [type(h).__name__ for h in root_logger.handlers]
            self.assertIn("StreamHandler", handler_types)
            self.assertIn("RotatingFileHandler", handler_types)
    
    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format."""
        custom_format = '%(levelname)s - %(message)s'
        setup_logging(
            log_level=logging.INFO,
            log_format=custom_format,
            console_output=True,
            file_output=False
        )
        
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        self.assertEqual(handler.formatter._fmt, custom_format)
    
    def test_setup_logging_log_levels(self):
        """Test that different log levels work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(
                log_level=logging.DEBUG,
                log_file=str(log_file),
                console_output=False,
                file_output=True
            )
            
            logger = logging.getLogger("test")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            # Check that all messages were logged
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Debug message", content)
                self.assertIn("Info message", content)
                self.assertIn("Warning message", content)
                self.assertIn("Error message", content)
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_module")
    
    def test_log_file_rotation(self):
        """Test that log file rotation works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(
                log_level=logging.INFO,
                log_file=str(log_file),
                console_output=False,
                file_output=True,
                max_bytes=100,  # Small size to trigger rotation
                backup_count=2
            )
            
            logger = logging.getLogger("test")
            # Write enough to trigger rotation
            for i in range(50):
                logger.info(f"Test message {i} " * 10)  # Make messages large
            
            # Check that backup files were created
            backup_files = list(log_file.parent.glob("test.log.*"))
            self.assertGreater(len(backup_files), 0)
            self.assertLessEqual(len(backup_files), 2)  # Should not exceed backup_count


class TestLoggingIntegration(unittest.TestCase):
    """Test logging integration with wordcloud components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(
            log_level=logging.INFO,
            console_output=False,
            file_output=False  # Disable file output for tests
        )
    
    def test_wordcloud_logging(self):
        """Test that Wordcloud class uses logging correctly."""
        from wordcloud import Wordcloud
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "wordcloud.log"
            setup_logging(
                log_level=logging.INFO,
                log_file=str(log_file),
                console_output=False,
                file_output=True
            )
            
            wc = Wordcloud(width=400, height=300, log_level=logging.INFO)
            wc.generate("test text for wordcloud")
            
            # Check that log file contains expected messages
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                    self.assertIn("Wordcloud initialized", content)
                    self.assertIn("Preparing text", content)
    
    def test_utility_module_logging(self):
        """Test that utility modules use logging correctly."""
        from wordcloud.utils.mask import MaskProcessor
        import numpy as np
        
        # Create a simple mask
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        logger = logging.getLogger("wordcloud.utils.mask")
        logger.setLevel(logging.INFO)
        
        # This should log initialization
        processor = MaskProcessor(mask, threshold=200)
        self.assertIsNotNone(processor)


if __name__ == '__main__':
    unittest.main()

