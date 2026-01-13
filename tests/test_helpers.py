"""
Tests for helper utility functions.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from wordcloud.utils.helpers import (
    normalize_path,
    ensure_parent_dir,
    create_folder,
    create_tracking_structure,
)


class TestNormalizePath(unittest.TestCase):
    """Tests for normalize_path function."""

    def test_normalize_string_path(self):
        """Test normalizing a string path."""
        result = normalize_path("some/path")
        self.assertIsInstance(result, Path)
        self.assertEqual(str(result), "some/path")

    def test_normalize_path_object(self):
        """Test normalizing a Path object."""
        path_obj = Path("some/path")
        result = normalize_path(path_obj)
        self.assertIsInstance(result, Path)
        self.assertEqual(result, path_obj)
        # Should return the same object (or equivalent)
        self.assertEqual(str(result), "some/path")


class TestEnsureParentDir(unittest.TestCase):
    """Tests for ensure_parent_dir function."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_parent_dir_creates_parent(self):
        """Test that ensure_parent_dir creates parent directory."""
        file_path = self.temp_dir / "subdir" / "file.txt"
        result = ensure_parent_dir(file_path)
        
        self.assertIsInstance(result, Path)
        self.assertEqual(result, file_path)
        self.assertTrue(file_path.parent.exists())
        self.assertTrue(file_path.parent.is_dir())

    def test_ensure_parent_dir_existing_parent(self):
        """Test ensure_parent_dir with existing parent directory."""
        parent_dir = self.temp_dir / "existing"
        parent_dir.mkdir()
        file_path = parent_dir / "file.txt"
        
        result = ensure_parent_dir(file_path)
        self.assertEqual(result, file_path)
        self.assertTrue(parent_dir.exists())

    def test_ensure_parent_dir_nested_paths(self):
        """Test ensure_parent_dir with deeply nested paths."""
        file_path = self.temp_dir / "a" / "b" / "c" / "d" / "file.txt"
        result = ensure_parent_dir(file_path)
        
        self.assertEqual(result, file_path)
        self.assertTrue((self.temp_dir / "a" / "b" / "c" / "d").exists())

    def test_ensure_parent_dir_string_input(self):
        """Test ensure_parent_dir accepts string input."""
        file_path = str(self.temp_dir / "subdir" / "file.txt")
        result = ensure_parent_dir(file_path)
        
        self.assertIsInstance(result, Path)
        self.assertTrue(Path(file_path).parent.exists())

    def test_ensure_parent_dir_root_path(self):
        """Test ensure_parent_dir with root path (no parent)."""
        # Root path has no parent, should not raise error
        root_path = Path("/")
        result = ensure_parent_dir(root_path)
        self.assertEqual(result, root_path)


class TestCreateFolder(unittest.TestCase):
    """Tests for create_folder function."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_create_new_folder(self):
        """Test creating a new folder."""
        folder_path = self.temp_dir / "new_folder"
        result = create_folder(folder_path)
        
        self.assertIsInstance(result, Path)
        self.assertEqual(result, folder_path)
        self.assertTrue(folder_path.exists())
        self.assertTrue(folder_path.is_dir())

    def test_create_existing_folder(self):
        """Test creating a folder that already exists."""
        folder_path = self.temp_dir / "existing_folder"
        folder_path.mkdir()
        
        result = create_folder(folder_path)
        self.assertEqual(result, folder_path)
        self.assertTrue(folder_path.exists())

    def test_create_nested_folders(self):
        """Test creating nested folder structure."""
        folder_path = self.temp_dir / "a" / "b" / "c"
        result = create_folder(folder_path, parents=True)
        
        self.assertEqual(result, folder_path)
        self.assertTrue(folder_path.exists())
        self.assertTrue((self.temp_dir / "a" / "b").exists())

    def test_create_folder_without_parents(self):
        """Test creating folder without creating parents."""
        parent = self.temp_dir / "parent"
        folder_path = parent / "child"
        
        # Should raise error if parents don't exist and parents=False
        with self.assertRaises(FileNotFoundError):
            create_folder(folder_path, parents=False)

    def test_create_folder_string_input(self):
        """Test create_folder accepts string input."""
        folder_path = str(self.temp_dir / "string_folder")
        result = create_folder(folder_path)
        
        self.assertIsInstance(result, Path)
        self.assertTrue(Path(folder_path).exists())

    @patch("wordcloud.utils.helpers.file_utils.logger")
    def test_create_folder_logs_debug(self, mock_logger):
        """Test that create_folder logs debug message."""
        folder_path = self.temp_dir / "logged_folder"
        create_folder(folder_path)
        
        # Check that debug was called
        mock_logger.debug.assert_called()
        self.assertIn("created/already exists", mock_logger.debug.call_args[0][0])

    @patch("wordcloud.utils.helpers.file_utils.logger")
    def test_create_folder_logs_error_on_oserror(self, mock_logger):
        """Test that create_folder logs error on OSError."""
        folder_path = self.temp_dir / "error_folder"
        
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            create_folder(folder_path)
            mock_logger.error.assert_called()


class TestCreateTrackingStructure(unittest.TestCase):
    """Tests for create_tracking_structure function."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_create_tracking_structure_new(self):
        """Test creating tracking structure from scratch."""
        base_dir = self.temp_dir / "Tracking"
        strategy = "random"
        
        result = create_tracking_structure(base_dir, strategy)
        
        self.assertIsInstance(result, Path)
        self.assertEqual(result, base_dir / strategy)
        self.assertTrue(base_dir.exists())
        self.assertTrue(result.exists())
        self.assertTrue(result.is_dir())

    def test_create_tracking_structure_existing_base(self):
        """Test creating tracking structure with existing base directory."""
        base_dir = self.temp_dir / "Tracking"
        base_dir.mkdir()
        strategy = "brute"
        
        result = create_tracking_structure(base_dir, strategy)
        
        self.assertEqual(result, base_dir / strategy)
        self.assertTrue(result.exists())

    def test_create_tracking_structure_existing_both(self):
        """Test creating tracking structure when both directories exist."""
        base_dir = self.temp_dir / "Tracking"
        strategy_dir = base_dir / "pytag"
        strategy_dir.mkdir(parents=True)
        
        result = create_tracking_structure(base_dir, "pytag")
        
        self.assertEqual(result, strategy_dir)
        self.assertTrue(result.exists())

    def test_create_tracking_structure_string_input(self):
        """Test create_tracking_structure accepts string input."""
        base_dir = str(self.temp_dir / "Tracking")
        strategy = "quad"
        
        result = create_tracking_structure(base_dir, strategy)
        
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())

    def test_create_tracking_structure_multiple_strategies(self):
        """Test creating tracking structure for multiple strategies."""
        base_dir = self.temp_dir / "Tracking"
        
        strategy1_dir = create_tracking_structure(base_dir, "strategy1")
        strategy2_dir = create_tracking_structure(base_dir, "strategy2")
        
        self.assertTrue(strategy1_dir.exists())
        self.assertTrue(strategy2_dir.exists())
        self.assertEqual(strategy1_dir.parent, strategy2_dir.parent)
        self.assertEqual(strategy1_dir.parent, base_dir)


if __name__ == "__main__":
    unittest.main()

