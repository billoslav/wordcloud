import unittest
from PIL import Image
from unittest.mock import patch, MagicMock, mock_open
from wordcloud import Wordcloud, IntegralImage
import unittest
import numpy as np
import os
import unittest
import numpy as np
from PIL import Image
import math
import os
from unittest.mock import patch, MagicMock

FONT_PATH = "fonts/Arial Unicode.ttf"

class TestIntegralImage(unittest.TestCase):

    def setUp(self):
        self.height = 100
        self.width = 150
        self.trace_margin = 200
        self.integral_image = IntegralImage(self.height, self.width)
        self.small_integral_image = IntegralImage(20, 20)
        self.tracking_integral_img = IntegralImage(self.height, self.width, tracing=True)
        self.free_locations = [(0,0),(1,1),(10,10),(10,20),(50,50),(100,50),(49,49),(149, 99),(63,85),(92,34)]
        self.rectangular_locations = [(0,0),(1,1),(10,10),(10,20),(4,4),(100,50),(49,49),(35,70),(70,40),(71,33)]
        self.rectangular_no_locations = [(9,9),(11,21),(100,50),(49,49)]
        self.archimedian_locations = [(0,0),(1,1),(10,10),(27,27),(6,87),(68,47),(49,49),(35,70),(70,40),(71,33)]
        self.archimedian_no_locations = [(9,9),(11,21),(100,50),(49,49)]
        self.KD_only_location = [(49,49)]
        
        self.test_image = np.ones((self.height, self.width), dtype=np.uint8)
        self.mock_image = MagicMock(spec=Image.Image)  # Mock PIL Image object
        
    def test_initialization(self):
        self.assertEqual(self.integral_image.height, self.height)
        self.assertEqual(self.integral_image.width, self.width)
        self.assertTrue(np.all(self.integral_image.integral == 0))

    def test_find_position_random(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="random")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_brute(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="brute")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_rectangular(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="rectangular")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)
    
    def test_find_position_rectangular_reverse(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="rectangular_reverse")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_archimedian(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="archimedian")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_archimedian_reverse(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="archimedian_reverse")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_kdtree(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="KDTree")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_quad(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="quad")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_pytag(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="pytag")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_pytag_reverse(self):
        size_x = 20
        size_y = 30
        position = self.integral_image.find_position(size_x, size_y, place_strategy="pytag_reverse")
        self.assertIsNotNone(position)
        x, y = position
        self.assertTrue(0 <= x <= self.width - size_x)
        self.assertTrue(0 <= y <= self.height - size_y)

    def test_find_position_non_existing_strategy(self):
        size_x = 20
        size_y = 30
        with self.assertRaises(ValueError):
            self.integral_image.find_position(size_x, size_y, place_strategy="non_existing")

    def test_find_position_no_free_locations(self):
        size_x = self.width  # Make size larger than available space
        size_y = self.height
        position = self.integral_image.find_position(size_x, size_y, place_strategy="random")
        self.assertIsNone(position)

    @patch("wordcloud.IntegralImage.is_valid_position")
    def test_find_position_basic(self, mock_is_valid_position):
        size_x = 20
        size_y = 30
        mock_is_valid_position.return_value = True  # Simulate a valid position
        result = self.integral_image.find_position(size_x, size_y, "random")
        self.assertIsNotNone(result)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)
        mock_is_valid_position.assert_called()

    @patch("wordcloud.IntegralImage.is_valid_position")
    def test_find_position_no_valid_position(self, mock_is_valid_position):
        size_x = 20
        size_y = 30
        mock_is_valid_position.return_value = False  # Simulate no valid position found
        result = self.integral_image.find_position(size_x, size_y, "random")
        self.assertIsNone(result)
        mock_is_valid_position.assert_called()

    @patch("wordcloud.IntegralImage.is_valid_position")
    def test_find_position_out_of_bounds(self, mock_is_valid_position):
        size_x = 40
        size_y = 40
        result = self.small_integral_image.find_position(size_x, size_y, "random")
        self.assertIsNone(result)
        mock_is_valid_position.assert_not_called()

    def test_find_position_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.integral_image.find_position(20, 30, place_strategy="invalid_strategy")

    @patch("wordcloud.IntegralImage.is_valid_position")
    def test_find_position_zero_size(self, mock_is_valid_position):
        size_x = 0
        size_y = 0
        mock_is_valid_position.return_value = True
        result = self.integral_image.find_position(size_x, size_y, "random")
        self.assertIsNotNone(result)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)
        mock_is_valid_position.assert_called()

    def test_find_position_negative_size(self):
        size_x = -10
        size_y = -20
        with self.assertRaises(ValueError):
            self.integral_image.find_position(size_x, size_y, "random")

    def test_update(self):
        x = 25
        y = 30
        self.assertTrue(np.all(self.integral_image.integral[:y, :x] == 0))  # Before update area should be unchanged.
        self.assertFalse(np.any(self.integral_image.integral[y:, x:] != 0))
        
        self.integral_image.update(self.test_image, x, y)

        self.assertFalse(self.integral_image.integral[y, x] == 0) # changed position
        self.assertTrue(np.any(self.integral_image.integral[y:, x:] != 0))  # After update area should be changed.

    def test_update_at_edge(self):
        x = self.integral_image.width - self.test_image.shape[1]
        y = self.integral_image.height - self.test_image.shape[0]
        self.integral_image.update(self.test_image, x, y)

        self.assertTrue(np.any(self.integral_image.integral[y:, x:] != 0))

    def test_update_with_existing_values(self):
        x = 25
        y = 30
        # Set some initial values in the integral image
        self.integral_image.integral[y:y+10, x:x+10] = 5

        self.integral_image.update(self.test_image, x, y)

        self.assertTrue(np.any(self.integral_image.integral[y:, x:] != 0))

    def test_update_empty_image(self):
        empty_image = np.zeros((self.height, self.width), dtype=np.uint8)
        x = 25
        y = 30
        self.integral_image.update(empty_image, x, y)
        self.assertTrue(np.all(self.integral_image.integral == 0)) # integral should be unchanged

    def test_update_larger_image(self):
        larger_image = np.ones((150, 200), dtype=np.uint8)  # Larger image
        x = 10
        y = 20
        with self.assertRaises(ValueError): #Expect value error
            self.integral_image.update(larger_image, x, y)

    def test_update_negative_coordinates(self):
      x = -10
      y = -20
      with self.assertRaises(ValueError):
            self.integral_image.update(self.test_image, x, y)

    def test_update_coordinates_out_of_bounds(self):
      x = self.integral_image.width + 10
      y = self.integral_image.height + 20
      self.integral_image.update(self.test_image, x, y)
      self.assertTrue(np.all(self.integral_image.integral == 0)) # integral should be unchanged

    def test_update_float_coordinates(self):
        x = 25.5
        y = 30.5
        
        with self.assertRaises(TypeError):
            self.integral_image.update(self.test_image, x, y)
            
    def test_update_valid(self):
        self.integral_image.update(self.test_image, 20, 30)
        self.assertEqual(self.integral_image.integral[30:40, 20:35].sum(), 6600)

    def test_update_out_of_bounds_x(self):
        self.integral_image.update(self.test_image, self.width + 5, 30)
        self.assertTrue(np.all(self.integral_image.integral == 0))

    def test_update_out_of_bounds_y(self):
        self.integral_image.update(self.test_image, 20, self.height + 5)
        self.assertTrue(np.all(self.integral_image.integral == 0))

    @patch('os.makedirs')
    def test_create_folder(self, mock_makedirs):
        folder_name = "test_folder"
        self.integral_image.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)

    @patch('os.makedirs')
    def test_create_folder_with_parent(self, mock_makedirs):
        folder_name = "test_folder"
        parent_name = "parent_folder"
        self.integral_image.create_folder(folder_name, parent_name)
        mock_makedirs.assert_called_once_with(os.path.join(parent_name, folder_name), exist_ok=True)

    @patch("os.makedirs")
    def test_create_folder_existing(self, mock_makedirs):
        folder_name = "test_folder"
        mock_makedirs.side_effect = FileExistsError  # Simulate existing folder
        self.integral_image.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)  # Still called with exist_ok=True

    @patch("os.makedirs")
    def test_create_folder_os_error(self, mock_makedirs):
        folder_name = "test_folder"
        mock_makedirs.side_effect = OSError("Simulated OSError")  # Simulate an OSError
        self.integral_image.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)

    @patch("os.makedirs")
    def test_create_folder_nested(self, mock_makedirs):
        folder_name = "path/to/folder"
        self.integral_image.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)

    @patch('os.makedirs')
    def test_create_tracking_structure(self, mock_makedirs):
        directory = "test_dir"
        place_strategy = "random"
        expected_path1 = directory
        expected_path2 = os.path.join(directory, place_strategy)

        result = self.integral_image.create_tracking_structure(directory, place_strategy)

        self.assertEqual(mock_makedirs.call_count, 2)
        mock_makedirs.assert_any_call(expected_path1, exist_ok=True)
        mock_makedirs.assert_any_call(expected_path2, exist_ok=True)
        self.assertEqual(result, f"{directory}/{place_strategy}/")

    @patch('os.makedirs')
    def test_create_tracking_structure_existing_folders(self, mock_makedirs):
        directory = "test_dir"
        place_strategy = "random"

        # Simulate existing directory
        mock_makedirs.side_effect = FileExistsError

        result = self.integral_image.create_tracking_structure(directory, place_strategy)

        self.assertEqual(mock_makedirs.call_count, 2)  # Still called twice
        self.assertEqual(result, f"{directory}/{place_strategy}/") # Path should still be returned

    @patch('os.makedirs')
    def test_create_tracking_structure_os_error(self, mock_makedirs):
      directory = "test_dir"
      place_strategy = "random"

      # Simulate OSError
      mock_makedirs.side_effect = OSError("Simulated OSError")

      result = self.integral_image.create_tracking_structure(directory, place_strategy)

      self.assertEqual(mock_makedirs.call_count, 2) # Still called twice
      self.assertEqual(result, f"{directory}/{place_strategy}/") # Path should still be returned

    @patch("os.makedirs")
    def test_create_tracking_structure_nested_directory(self, mock_makedirs):
        directory = "path/to/test_dir"
        place_strategy = "random"
        expected_path1 = directory
        expected_path2 = os.path.join(directory, place_strategy)

        result = self.integral_image.create_tracking_structure(directory, place_strategy)

        self.assertEqual(mock_makedirs.call_count, 2)
        mock_makedirs.assert_any_call(expected_path1, exist_ok=True)
        mock_makedirs.assert_any_call(expected_path2, exist_ok=True)
        self.assertEqual(result, f"{directory}/{place_strategy}/")

    def test_create_tracking_structure_empty_strategy(self):
      directory = "test_dir"
      place_strategy = ""

      result = self.integral_image.create_tracking_structure(directory, place_strategy)

      self.assertEqual(result, f"{directory}/{place_strategy}/")

    def test_is_valid_position(self):
        size_x = 20
        size_y = 30
        # Initially all zeros, so any position within bounds should be valid
        self.assertTrue(self.integral_image.is_valid_position(0, 0, size_y, size_x))  # Accessing private method for testing

        # Simulate occupied area (set some integral values)
        self.integral_image.integral[10:20, 10:30] = 1  # Simulate an occupied area
        self.assertFalse(self.integral_image.is_valid_position(10, 10, size_y, size_x))  # Accessing private method for testing
        self.assertTrue(self.integral_image.is_valid_position(0, 0, size_y, size_x)) # Check another position

    def test_is_valid_position_basic(self):
        size_x = 20
        size_y = 30
        x = 10
        y = 20

        # Simulate an empty area (all zeros)
        self.integral_image.integral[:] = 0
        self.assertTrue(self.integral_image.is_valid_position(y, x, size_y, size_x))

        # Simulate an occupied area
        self.integral_image.integral[y:y+size_y, x:x+size_x] = 1
        self.assertFalse(self.integral_image.is_valid_position(y, x, size_y, size_x))

    def test_is_valid_position_at_edge(self):
        size_x = 20
        size_y = 30
        x = self.integral_image.width - size_x -1
        y = self.integral_image.height - size_y -1

        self.integral_image.integral[:] = 0
        self.assertTrue(self.integral_image.is_valid_position(y, x, size_y, size_x))
        
    def test_is_valid_position_edge_out_of_bounds(self):
        size_x = 20
        size_y = 30
        x = self.integral_image.width - size_x
        y = self.integral_image.height - size_y

        self.integral_image.integral[:] = 0
        with self.assertRaises(IndexError):
            self.integral_image.is_valid_position(y, x, size_y, size_x)

    def test_is_valid_position_out_of_bounds_basic(self):
        size_x = 20
        size_y = 30
        x = self.integral_image.width + 10
        y = self.integral_image.height + 10

        self.integral_image.integral[:] = 0
        with self.assertRaises(IndexError):
            self.integral_image.is_valid_position(y, x, size_y, size_x)

    def test_is_valid_position_negative_coordinates(self):
        size_x = 20
        size_y = 30
        x = -10
        y = -10

        self.integral_image.integral[:] = 0
        with self.assertRaises(ValueError):
            self.integral_image.is_valid_position(y, x, size_y, size_x)
            
    def test_is_valid_position_negative_size(self):
        size_x = -20
        size_y = -30
        x = 10
        y = 10

        self.integral_image.integral[:] = 0
        with self.assertRaises(ValueError):
            self.integral_image.is_valid_position(y, x, size_y, size_x)
            
    def test_is_valid_position_empty(self):
        self.assertTrue(self.integral_image.is_valid_position(0, 0, 10, 10))

    def test_is_valid_position_occupied(self):
        self.integral_image.update(np.ones((100, 150), dtype=np.uint64), 0, 0)
        self.assertFalse(self.integral_image.is_valid_position(0, 0, 10, 10))

    def test_is_valid_position_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.integral_image.is_valid_position(-1, 0, 10, 10)
        with self.assertRaises(ValueError):
            self.integral_image.is_valid_position(0, -1, 10, 10)
        with self.assertRaises(ValueError):
            self.integral_image.is_valid_position(0, 0, -10, 10)
        with self.assertRaises(ValueError):
            self.integral_image.is_valid_position(0, 0, 10, -10)

    def test_check_bounds(self):
        size_x = 20
        size_y = 30
        
        self.assertFalse(self.integral_image.check_bounds(10, 10, size_x, size_y))
        self.assertFalse(self.integral_image.check_bounds(-1, 10, size_x, size_y))
        self.assertFalse(self.integral_image.check_bounds(10, -1, size_x, size_y))
        self.assertFalse(self.integral_image.check_bounds(self.width + 1, 10, size_x, size_y))
        self.assertFalse(self.integral_image.check_bounds(10, self.height + 1, size_x, size_y))
        self.assertTrue(self.integral_image.check_bounds(-1, -1, size_x, size_y))
        self.assertTrue(self.integral_image.check_bounds(self.width + 1, self.height + 1, size_x, size_y))
        self.assertTrue(self.integral_image.check_bounds(self.width + 1, -1, size_x, size_y))
        self.assertTrue(self.integral_image.check_bounds(-1, self.height + 1, size_x, size_y))
        self.assertTrue(self.integral_image.check_bounds(10, 10, self.width, self.height))
        self.assertFalse(self.integral_image.check_bounds(-1, 10, self.width, size_y))
        self.assertTrue(self.integral_image.check_bounds(10, -1, self.width, size_y))
        self.assertTrue(self.integral_image.check_bounds(-1, 10, size_x, self.height))
        self.assertFalse(self.integral_image.check_bounds(10, -1, size_x, self.height))
        self.assertTrue(self.integral_image.check_bounds(self.width + 1, self.height + 1, size_x, size_y))
        self.assertTrue(self.integral_image.check_bounds(self.width + 1, self.height + 1, size_x, size_y))
        self.assertTrue(self.integral_image.check_bounds(self.width + 1, self.height + 1, self.width, self.height))
        
    def test_check_bounds_within_bounds(self):
        size_x = 20
        size_y = 30
        x = 10
        y = 20
        self.assertFalse(self.integral_image.check_bounds(x, y, size_x, size_y))

    def test_check_bounds_only_x_out_of_bounds(self):
        size_x = 20
        size_y = 30
        x = self.integral_image.width + 10
        y = 20
        self.assertFalse(self.integral_image.check_bounds(x, y, size_x, size_y))  # x out of bounds

    def test_check_bounds_only_y_out_of_bounds(self):
        size_x = 20
        size_y = 30
        x = 10
        y = self.integral_image.height + 10
        self.assertFalse(self.integral_image.check_bounds(x, y, size_x, size_y))  # y out of bounds

    def test_check_bounds_both_out_of_bounds_positive(self):
        size_x = 20
        size_y = 30
        x = self.integral_image.width + 10
        y = self.integral_image.height + 10
        self.assertTrue(self.integral_image.check_bounds(x, y, size_x, size_y))  # Both out of bounds

    def test_check_bounds_x_out_of_bounds_negative(self):
        size_x = 20
        size_y = 30
        x = -10
        y = 20
        self.assertFalse(self.integral_image.check_bounds(x, y, size_x, size_y))  # x out of bounds

    def test_check_bounds_y_out_of_bounds_negative(self):
        size_x = 20
        size_y = 30
        x = 10
        y = -10
        self.assertFalse(self.integral_image.check_bounds(x, y, size_x, size_y))  # y out of bounds

    def test_check_bounds_both_out_of_bounds_negative(self):
        size_x = 20
        size_y = 30
        x = -10
        y = -10
        self.assertTrue(self.integral_image.check_bounds(x, y, size_x, size_y))  # Both out of bounds

    def test_check_bounds_zero_size(self):
        size_x = 0
        size_y = 0
        x = 10
        y = 20
        self.assertFalse(self.integral_image.check_bounds(x, y, size_x, size_y))

    def test_check_bounds_negative_size(self):
        size_x = -10
        size_y = -20
        x = 10
        y = 20
        self.assertFalse(self.integral_image.check_bounds(x, y, size_x, size_y))
        
    def test_check_bounds_inside(self):
        self.assertFalse(self.integral_image.check_bounds(10, 10, 20, 20))

    def test_check_bounds_right_edge(self):
        self.assertFalse(self.integral_image.check_bounds(self.width - 10, 10, 20, 20))

    def test_check_bounds_bottom_edge(self):
        self.assertFalse(self.integral_image.check_bounds(10, self.height - 10, 20, 20))

    def test_check_bounds_top_edge(self):
        self.assertFalse(self.integral_image.check_bounds(10, -5, 20, 20))

    def test_check_bounds_left_edge(self):
        self.assertFalse(self.integral_image.check_bounds(-5, 10, 20, 20))
        
    #TODO - how to test Draw trace point
    def test_draw_trace_point_and_save_trace_img(self):
        self.tracking_integral_img.directory_name = "Tracing_test"
        self.tracking_integral_img.tracing_setup(10, 20, "rectangular", "test_rectangular")

        self.tracking_integral_img.draw_trace_point(10, 20)
        self.tracking_integral_img.save_trace_img()
        
        self.assertTrue(os.path.exists(self.tracking_integral_img.trace_img_name))
        os.remove(self.tracking_integral_img.trace_img_name)
        os.rmdir("Tracing_test/rectangular/")
        os.rmdir("Tracing_test/")
    
    #TODO - how to test Save trace IMG
    
    def test_random_function_basic(self):
        size_x = 20
        size_y = 30
        result = self.integral_image.random(self.free_locations, self.width, self.height, size_x, size_y)
        x, y = result
        self.assertIn((x, y), self.free_locations)
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_random_function_multiple_calls(self):
        size_x = 20
        size_y = 30
        results = []
        for _ in range(10):
            result = self.integral_image.rectangular_code(self.free_locations, 0, 0, size_x, size_y)
            if result:
                results.append(result)
                
        self.assertEqual(len(results), 10)

    def test_random_function_zero_size(self):
        size_x = 0
        size_y = 0
        result = self.integral_image.random(self.free_locations, self.width, self.height, size_x, size_y)
        x, y = result
        self.assertIn((x, y), self.free_locations)
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_random_function_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        result = self.integral_image.random(self.free_locations, self.width, self.height, size_x, size_y)
        x, y = result
        self.assertIn((x, y), self.free_locations)
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_rectangular_code_basic(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.rectangular_code(self.rectangular_locations, width_x, height_y, size_x, size_y)
        self.assertIsNotNone(result)

        x, y = result
        self.assertIn((x, y), self.rectangular_locations)
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_rectangular_code_multiple_calls(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        results = set()
        for _ in range(10):
            result = self.integral_image.rectangular_code(self.rectangular_locations, width_x, height_y, size_x, size_y)
            if result:
                results.add(result)
        
        self.assertTrue(len(results) > 0)

    def test_rectangular_code_zero_size(self):
        size_x = 0
        size_y = 0
        result = self.integral_image.rectangular_code(self.rectangular_locations, 0, 0, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        
    def test_rectangular_code_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        result = self.integral_image.rectangular_code(self.free_locations, 0, 0, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)
        
    def test_rectangular_code_reverse_basic(self):
        size_x = 10
        size_y = 20
        
        result = self.integral_image.rectangular_code(self.rectangular_locations, 0, 0, size_x, size_y, reverse=True)
        self.assertIsNotNone(result)

        x, y = result
        self.assertIn((x, y), self.rectangular_locations)
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_rectangular_code_reverse_multiple_calls(self):
        size_x = 10
        size_y = 20
        
        results = set()
        for _ in range(10):
            result = self.integral_image.rectangular_code(self.rectangular_locations, 0, 0, size_x, size_y, reverse=True)
            if result:
                results.add(result)
        
        self.assertTrue(len(results) > 0)

    def test_rectangular_code_reverse_zero_size(self):
        size_x = 0
        size_y = 0
        result = self.integral_image.rectangular_code(self.rectangular_locations, 0, 0, size_x, size_y, reverse=True)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        
    def test_rectangular_code_reverse_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        result = self.integral_image.rectangular_code(self.rectangular_locations, 0, 0, size_x, size_y, reverse=True)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)
        
    def test_rectangular_basic(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.rectangular(self.rectangular_locations, width_x, height_y, size_x, size_y)
        self.assertIsNotNone(result)

        x, y = result
        self.assertIn((x, y), self.rectangular_locations)
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_rectangular_multiple_calls(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        results = set()
        for _ in range(10):
            result = self.integral_image.rectangular(self.rectangular_locations, width_x, height_y, size_x, size_y)
            if result:
                results.add(result)

        self.assertTrue(len(results) > 0)

    def test_rectangular_zero_size(self):
        size_x = 0
        size_y = 0
        result = self.integral_image.rectangular(self.rectangular_locations, 0, 0, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        
    def test_rectangular_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        result = self.integral_image.rectangular(self.free_locations, 0, 0, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)
        
    def test_rectangular_reverse_basic(self):
        size_x = 10
        size_y = 20
        
        result = self.integral_image.rectangular_reverse(self.rectangular_locations, 0, 0, size_x, size_y)
        self.assertIsNotNone(result)

        x, y = result
        self.assertIn((x, y), self.rectangular_locations)
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_rectangular_reverse_multiple_calls(self):
        size_x = 10
        size_y = 20
        
        results = set()
        for _ in range(10):
            result = self.integral_image.rectangular_reverse(self.rectangular_locations, 0, 0, size_x, size_y)
            if result:
                results.add(result)

        self.assertTrue(len(results) > 0)

    def test_rectangular_reverse_zero_size(self):
        size_x = 0
        size_y = 0
        result = self.integral_image.rectangular_reverse(self.rectangular_locations, 0, 0, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        
    def test_rectangular_reverse_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        result = self.integral_image.rectangular_reverse(self.rectangular_locations, 0, 0, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)
        
    def test_rectangular_code_no_location(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.rectangular_code(self.rectangular_no_locations, width_x, height_y, size_x, size_y)
        self.assertIsNone(result)
    
    def test_rectangular_code_reverse_no_location(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.rectangular_code(self.rectangular_no_locations, width_x, height_y, size_x, size_y, reverse=True)
        self.assertIsNone(result)
    
    def test_rectangular_no_location(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.rectangular(self.rectangular_no_locations, width_x, height_y, size_x, size_y)
        self.assertIsNone(result)
        
    def test_rectangular_reverse_no_location(self):
        size_x = 10
        size_y = 20
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.rectangular_reverse(self.rectangular_no_locations, width_x, height_y, size_x, size_y)
        self.assertIsNone(result)

    def test_archimedian_basic(self):
        size_x = 20
        size_y = 30
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.archimedian(self.archimedian_locations, width_x, height_y, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_archimedian_multiple_calls(self):
        size_x = 20
        size_y = 30
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        results = []
        for _ in range(10):
            result = self.integral_image.archimedian(self.archimedian_locations, width_x, height_y, size_x, size_y)
            if result:
                results.append(result)
                
        self.assertEqual(len(results), 10)

    def test_archimedian_zero_size(self):
        size_x = 0
        size_y = 0
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.archimedian(self.archimedian_locations, width_x, height_y, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_archimedian_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.archimedian(self.archimedian_locations, width_x, height_y, size_x, size_y)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_archimedian_spiral_movement(self):
        size_x = 20
        size_y = 30
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        results = []
        for _ in range(20):
            result = self.integral_image.archimedian(self.archimedian_locations, width_x, height_y, size_x, size_y)
            if result:
                results.append(result)
                
        distances_from_center = [math.sqrt((x - width_x)**2 + (y - height_y)**2) for x, y in results]
        # Check if distances tend to increase (basic spiral movement check)
        for i in range(len(distances_from_center) - 1):
            self.assertTrue(distances_from_center[i+1] >= distances_from_center[i])

    def test_archimedian_no_location(self):
        size_x = 20
        size_y = 30
        height_y = (self.integral_image.height - size_y) // 2
        width_x = (self.integral_image.width - size_x) // 2
        
        result = self.integral_image.archimedian(self.archimedian_no_locations, width_x, height_y, size_x, size_y)
        self.assertIsNone(result)

    def test_archimedian_reverse_basic(self):
        size_x = 20
        size_y = 30
        result = self.integral_image.archimedian_reverse(self.archimedian_locations, self.integral_image.width, self.integral_image.height, 0, 0)
        self.assertIsNotNone(result)
        
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_archimedian_reverse_multiple_calls(self):
        results = []
        for _ in range(10):
            result = self.integral_image.archimedian_reverse(self.archimedian_locations, self.integral_image.width, self.integral_image.height, 0, 0)
            if result:
                results.append(result)
        
        self.assertGreater(len(set(results)), 0)

    def test_archimedian_reverse_zero_size(self):
        result = self.integral_image.archimedian_reverse(self.archimedian_locations, self.integral_image.width, self.integral_image.height, 0, 0)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_archimedian_reverse_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        result = self.integral_image.archimedian_reverse(self.archimedian_locations, self.integral_image.width, self.integral_image.height, 0, 0)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_archimedian_reverse_spiral_movement(self):
        center_x = self.integral_image.width // 2
        center_y = self.integral_image.height // 2
        results = [self.integral_image.archimedian_reverse(self.archimedian_locations, self.integral_image.width, self.integral_image.height, 0, 0) for _ in range(20)]

        distances_from_center = [math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in results]
        # Check if distances tend to decrease (basic reverse spiral movement check)
        for i in range(len(distances_from_center) - 1):
            self.assertTrue(distances_from_center[i+1] <= distances_from_center[i])
    
    def test_archimedian_reverse_no_location(self):
        size_x = 20
        size_y = 30
        result = self.integral_image.archimedian_reverse(self.archimedian_no_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        self.assertIsNone(result)

    def test_kdtree_basic(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.KDTree(self.free_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_kdtree_no_neighbors(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.KDTree([(size_x, size_y)], self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(x == size_x)
        self.assertTrue(y == size_y)

    def test_kdtree_empty_points(self):
        size_x = 20
        size_y = 30
        result = self.integral_image.KDTree([], self.integral_image.width, self.integral_image.height, size_x, size_y)
        self.assertIsNone(result)

    def test_kdtree_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2

        result = self.integral_image.KDTree(self.free_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_quad_basic(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        
        result = self.integral_image.quad(self.free_locations, size_x, size_y, 0, 0)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_quad_no_valid_position(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        
        result = self.integral_image.quad([], size_x, size_y, 0, 0)
        self.assertIsNone(result)
        
    def test_quad_large_size(self):
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        
        result = self.integral_image.quad(self.free_locations, size_x, size_y, 0, 0)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_pytag_code_basic(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag_code(self.free_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_pytag_code_no_valid_position(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag_code(self.KD_only_location, self.integral_image.width, self.integral_image.height, size_x, size_y)
        self.assertIsNone(result)

    def test_pytag_code_zero_size(self):
        # size should not play any role
        size_x = 0
        size_y = 0
        
        result = self.integral_image.pytag_code(self.free_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_pytag_code_large_size(self):
        # size should not play any role
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        
        result = self.integral_image.pytag_code(self.free_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)
        
    def test_pytag_code_reverse_basic(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag_code(self.rectangular_locations, self.integral_image.width, self.integral_image.height, size_x, size_y, is_reverse=True)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)

    def test_pytag_code_reverse_no_valid_position(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag_code(self.KD_only_location, self.integral_image.width, self.integral_image.height, size_x, size_y, is_reverse=True)
        self.assertIsNone(result)

    def test_pytag_code_reverse_zero_size(self):
        # size should not play any role
        size_x = 0
        size_y = 0
        
        result = self.integral_image.pytag_code(self.rectangular_locations, self.integral_image.width, self.integral_image.height, size_x, size_y, is_reverse=True)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)

    def test_pytag_code_reverse_large_size(self):
        # size should not play any role
        size_x = self.integral_image.width // 2
        size_y = self.integral_image.height // 2
        
        result = self.integral_image.pytag_code(self.rectangular_locations, self.integral_image.width, self.integral_image.height, size_x, size_y, is_reverse=True)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width)
        self.assertTrue(0 <= y <= self.integral_image.height)
        
    def test_pytag_basic(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag(self.free_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)
    
    def test_pytag_reverse_basic(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag_reverse(self.rectangular_locations, self.integral_image.width, self.integral_image.height, size_x, size_y)
        x, y = result
        self.assertTrue(0 <= x <= self.integral_image.width - size_x)
        self.assertTrue(0 <= y <= self.integral_image.height - size_y)
    
    def test_pytag_no_valid_position(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag(self.KD_only_location, self.integral_image.width, self.integral_image.height, size_x, size_y)
        self.assertIsNone(result)
        
    def test_pytag_reverse_no_valid_position(self):
        size_x = 20
        size_y = 30
        
        result = self.integral_image.pytag_reverse(self.KD_only_location, self.integral_image.width, self.integral_image.height, size_x, size_y)
        self.assertIsNone(result)

    def test_getitem_valid_indices(self):
        self.integral_image.update(self.test_image, 25, 30)
        value = self.integral_image.integral[35, 40]  # Access a value within the updated area
        self.assertNotEqual(value, 0)

    def test_getitem_invalid_indices(self):
        with self.assertRaises(IndexError):
            _ = self.integral_image.integral[0, self.integral_image.width + 1]  # Out of bounds

    def test_getitem_slices(self):
        self.integral_image.update(self.test_image, 25, 30)
        slice_result = self.integral_image.integral[30:40, 35:45]
        self.assertEqual(slice_result.shape, (10, 10))

    def test_lookup_edge_cases(self):
        self.integral_image.update(self.test_image, 25, 30)
        # Lookups at the edges of the updated area
        self.assertNotEqual(self.integral_image.integral[30, 25], 0)
        self.assertNotEqual(self.integral_image.integral[90, 75], 0)

    def test_lookup_specific_values(self):
        self.integral_image.update(self.test_image, 0, 0)
        self.assertEqual(self.integral_image.integral[0, 0], 1)
        self.assertEqual(self.integral_image.integral[1, 0], 2)
        self.assertEqual(self.integral_image.integral[0, 1], 2)
        self.assertEqual(self.integral_image.integral[1, 1], 4)

    def test_update_float_image(self):
        float_image = np.ones((self.height, self.width), dtype=np.float32)
        self.integral_image.update(float_image, 25, 30)
        self.assertTrue(np.any(self.integral_image.integral != 0))

    def test_update_color_image(self):
        color_image = np.ones((50, 60, 3), dtype=np.uint8)  # 3 color channels
        with self.assertRaises(ValueError): #Or TypeError, depends on your implementation
            self.integral_image.update(color_image, 25, 30)

    def test_find_position_small_sizes(self):
        result = self.integral_image.find_position(1, 1, "random")
        self.assertIsNotNone(result)

    #TODO tracing_setup testing
    def test_create_folder_exists(self):
        folder_name = "test_folder"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.integral_image.create_folder(folder_name)
        self.assertTrue(os.path.exists(folder_name))
        os.rmdir(folder_name)

    def test_create_folder_non_existent(self):
        folder_name = "new_test_folder"
        self.integral_image.create_folder(folder_name)
        self.assertTrue(os.path.exists(folder_name))
        os.rmdir(folder_name)

    def test_create_tracking_structure(self):
        directory = "test_track"
        strategy = "test_strat"
        path = self.integral_image.create_tracking_structure(directory, strategy)
        expected_path = f"{directory}/{strategy}/"
        self.assertTrue(os.path.exists(expected_path))
        self.assertEqual(path, expected_path)
        os.rmdir(expected_path)
        os.rmdir(directory)

    def test_tracing_setup(self):
        """Test the tracing_setup method with new tracking structure"""
        # Create a tracing-enabled integral image
        integral_image = IntegralImage(100, 150, tracing=True)
        integral_image.directory_name = "Tracing_test"
        
        # Call tracing_setup for the first time (should create new structure)
        integral_image.structure_created = False
        integral_image.tracing_setup(10, 20, "test_strategy", "test_word")
        
        # Check if the structure was created
        self.assertTrue(integral_image.structure_created)
        self.assertTrue(os.path.exists("Tracing_test/test_strategy"))
        
        # Check if the tracking image was created with correct dimensions
        self.assertEqual(integral_image.tracking_img.size, (integral_image.width + integral_image.trace_margin, 
                                                            integral_image.height + integral_image.trace_margin))
        
        # Check if trace_img_name was correctly set
        self.assertTrue("Tracing_test/test_strategy/tracing-test_word" in integral_image.trace_img_name)
        
        # Cleanup - handle potential errors
        try:
            # First remove any files that might have been created
            if os.path.exists(integral_image.trace_img_name):
                os.remove(integral_image.trace_img_name)
            
            # Remove the directories
            if os.path.exists("Tracing_test/test_strategy"):
                # List all files and remove them first
                for file in os.listdir("Tracing_test/test_strategy"):
                    file_path = os.path.join("Tracing_test/test_strategy", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir("Tracing_test/test_strategy")
            
            if os.path.exists("Tracing_test"):
                # List all subdirectories and remove them first
                for dir_name in os.listdir("Tracing_test"):
                    dir_path = os.path.join("Tracing_test", dir_name)
                    if os.path.isdir(dir_path):
                        # Remove any files in the directory
                        for file in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        os.rmdir(dir_path)
                os.rmdir("Tracing_test")
        except OSError:
            pass  # Ignore cleanup errors
        
    def test_tracing_setup_existing_structure(self):
        """Test tracing_setup method with an existing structure"""
        # Create a tracing-enabled integral image
        integral_image = IntegralImage(100, 150, tracing=True)
        integral_image.directory_name = "Tracing_test"
        
        # Create the directory structure manually
        if not os.path.exists("Tracing_test"):
            os.makedirs("Tracing_test")
        if not os.path.exists("Tracing_test/test_strategy"):
            os.makedirs("Tracing_test/test_strategy")
        
        # Call tracing_setup with existing structure
        integral_image.structure_created = True
        integral_image.tracing_setup(10, 20, "test_strategy", "test_word2")
        
        # Check if tracking image and draw object were created
        self.assertIsNotNone(integral_image.tracking_img)
        self.assertIsNotNone(integral_image.track_draw)
        
        # Check if trace_img_name contains the right path and word
        expected_path = "Tracing_test/test_strategy/tracing-test_word2"
        self.assertTrue(expected_path in integral_image.trace_img_name)
        
        # Cleanup - handle potential errors
        try:
            # First remove any files that might have been created
            if os.path.exists(integral_image.trace_img_name):
                os.remove(integral_image.trace_img_name)
            
            # Remove the directories
            if os.path.exists("Tracing_test/test_strategy"):
                # List all files and remove them first
                for file in os.listdir("Tracing_test/test_strategy"):
                    file_path = os.path.join("Tracing_test/test_strategy", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir("Tracing_test/test_strategy")
            
            if os.path.exists("Tracing_test"):
                # List all subdirectories and remove them first
                for dir_name in os.listdir("Tracing_test"):
                    dir_path = os.path.join("Tracing_test", dir_name)
                    if os.path.isdir(dir_path):
                        # Remove any files in the directory
                        for file in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        os.rmdir(dir_path)
                os.rmdir("Tracing_test")
        except OSError:
            pass  # Ignore cleanup errors
    
    def test_save_trace_img(self):
        """Test the save_trace_img method"""
        # Create a tracing-enabled integral image
        integral_image = IntegralImage(100, 150, tracing=True)
        integral_image.directory_name = "Tracing_test"
        
        # Setup tracing environment
        integral_image.tracing_setup(10, 20, "test_save_img", "test_word")
        
        # Save the trace image
        integral_image.save_trace_img()
        
        # Check if file was created
        self.assertTrue(os.path.exists(integral_image.trace_img_name))
        
        # Cleanup - handle potential errors
        try:
            # First remove any files that might have been created
            if os.path.exists(integral_image.trace_img_name):
                os.remove(integral_image.trace_img_name)
            
            # Remove the directories
            if os.path.exists("Tracing_test/test_save_img"):
                # List all files and remove them first
                for file in os.listdir("Tracing_test/test_save_img"):
                    file_path = os.path.join("Tracing_test/test_save_img", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir("Tracing_test/test_save_img")
            
            if os.path.exists("Tracing_test"):
                # List all subdirectories and remove them first
                for dir_name in os.listdir("Tracing_test"):
                    dir_path = os.path.join("Tracing_test", dir_name)
                    if os.path.isdir(dir_path):
                        # Remove any files in the directory
                        for file in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        os.rmdir(dir_path)
                os.rmdir("Tracing_test")
        except OSError:
            pass  # Ignore cleanup errors

    def test_draw_trace_point(self):
        """Test the draw_trace_point method"""
        # Create a tracing-enabled integral image
        integral_image = IntegralImage(100, 150, tracing=True)
        integral_image.directory_name = "Tracing_test"
        
        # Setup tracing environment
        integral_image.tracing_setup(10, 20, "test_trace_point", "test_word")
        
        # Draw a trace point
        test_x, test_y = 25, 30
        integral_image.draw_trace_point(test_x, test_y)
        
        # The exact color value can vary based on implementation
        # For "red" in L mode (grayscale), it could be any gray value
        # So we'll just check if the pixel has been modified from white (255)
        expected_point = (test_x + integral_image.half_trace_margin, 
                          test_y + integral_image.half_trace_margin)
        pixel_value = integral_image.tracking_img.getpixel(expected_point)
        self.assertNotEqual(pixel_value, 255)  # Should be modified from white
        
        # Cleanup - remove directories if they exist
        try:
            if os.path.exists(integral_image.trace_img_name):
                os.remove(integral_image.trace_img_name)
            if os.path.exists("Tracing_test/test_trace_point"):
                os.rmdir("Tracing_test/test_trace_point")
            if os.path.exists("Tracing_test"):
                os.rmdir("Tracing_test")
        except OSError:
            pass  # Ignore cleanup errors

    def test_save_trace_img(self):
        """Test the save_trace_img method"""
        # Create a tracing-enabled integral image
        integral_image = IntegralImage(100, 150, tracing=True)
        integral_image.directory_name = "Tracing_test"
        
        # Setup tracing environment
        integral_image.tracing_setup(10, 20, "test_save_img", "test_word")
        
        # Save the trace image
        integral_image.save_trace_img()
        
        # Check if file was created
        self.assertTrue(os.path.exists(integral_image.trace_img_name))
        
        # Cleanup - handle potential errors
        try:
            # First remove any files that might have been created
            if os.path.exists(integral_image.trace_img_name):
                os.remove(integral_image.trace_img_name)
            
            # Remove the directories
            if os.path.exists("Tracing_test/test_save_img"):
                # List all files and remove them first
                for file in os.listdir("Tracing_test/test_save_img"):
                    file_path = os.path.join("Tracing_test/test_save_img", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir("Tracing_test/test_save_img")
            
            if os.path.exists("Tracing_test"):
                # List all subdirectories and remove them first
                for dir_name in os.listdir("Tracing_test"):
                    dir_path = os.path.join("Tracing_test", dir_name)
                    if os.path.isdir(dir_path):
                        # Remove any files in the directory
                        for file in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        os.rmdir(dir_path)
                os.rmdir("Tracing_test")
        except OSError:
            pass  # Ignore cleanup errors
    
    def test_draw_trace_point_no_tracing(self):
        """Test draw_trace_point with tracing disabled"""
        # Create an integral image with tracing disabled
        integral_image = IntegralImage(100, 150, tracing=False)
        
        # Drawing a point should not raise an error even without tracing setup
        integral_image.draw_trace_point(25, 30)
        
    def test_save_trace_img_no_tracing(self):
        """Test save_trace_img with tracing disabled"""
        # Create an integral image with tracing disabled
        integral_image = IntegralImage(100, 150, tracing=False)
        
        # Saving should not raise an error even without tracing setup
        integral_image.save_trace_img()

class TestWordcloud(unittest.TestCase):

    def setUp(self):
        self.wc = Wordcloud(width=400, height=300, font_path=FONT_PATH,
                            stopwords=["but", "and", "are", "in", "the", "sit", "leo", "vel", "nec", "erat", "nulla", "non", "lorem", "ipsum", "Lorem"],
                            min_font_size=10)
        self.wc_small = Wordcloud(width=1, height=1, font_path=FONT_PATH,
                            stopwords=["but", "and", "are", "vel", "nec", "erat", "nulla", "non", "lorem", "ipsum", "Lorem"],
                            min_font_size=10)
        self.initial_positions = [
            (("test1", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red"),
            (("test2", 0.3, 1), "fonts/Cabal.ttf", 15, (100, 100), None, "blue"),
            (("test3", 0.7, 3), FONT_PATH, 25, (150, 150), None, "green")
        ]
        self.update_KDTree_positions = [
            (("test1", 0.5, 2), FONT_PATH, 30, (167, 139), None, "red"),
            (("test2", 0.3, 1), FONT_PATH, 18, (180, 161), None, "blue"),
            (("test3", 0.7, 3), FONT_PATH, 18, (180, 125), None, "green")
        ]

    def test_split_text_basic(self):
        text = "This isa test string."
        expected = {'this': 1, 'isa': 1, 'test': 1, 'string': 1}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_repeated_words(self):
        text = "This is a test string. This string has some repeated words. This."
        expected = {'this': 3, 'test': 1, 'string': 2, 'has': 1, 'some': 1, 'repeated': 1, 'words': 1}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_case_insensitive(self):
        text = "This isa Test string. this STRING"  # Mixed case
        expected = {'this': 2, 'isa': 1, 'test': 1, 'string': 2}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_punctuation(self):
        text = "This isa test string! This string, has some. words."
        expected = {'this': 2, 'isa': 1, 'test': 1, 'string': 2, 'has': 1, 'some': 1, 'words': 1}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_numbers_and_special_chars(self):
        text = "This isa test string 123.  string #$%^&*"
        expected = {'this': 1, 'isa': 1, 'test': 1, 'string': 2}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_empty_string(self):
        text = ""
        expected = {}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_stopwords(self):
        self.wc.stopwords = ["isa", "test"]  # Set stopwords
        text = "This isa test string. This string has some words."
        expected = {'this': 2, 'string': 2, 'has': 1, 'some': 1, 'words': 1}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_min_word_length(self):
        self.wc.min_word_length = 5  # Set minimum word length
        text = "This is a test string.  string has some words."
        expected = {'string': 2, 'words': 1}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_stopwords_and_min_word_length(self):
        self.wc.stopwords = ["isa", "words"]
        self.wc.min_word_length = 5
        text = "This isa test string. string has some words."
        expected = {'string': 2}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_long_text(self):  # Test with a longer text
        text = " ".join(["word"] * 1000)  # Create a string with 1000 "word"s
        expected = {'word': 1000}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_unicode(self):
        text = "This isa test strîng.  strüng has sôme wörds."  # Unicode characters
        expected = {'has': 1, 'isa': 1, 'strîng': 1, 'strüng': 1, 'sôme': 1, 'test': 1, 'this': 1, 'wörds': 1}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_hyphens(self):
      text = "This isa test hyphenated-word.  hyphenated-word has words."
      expected = {'this': 1, 'isa': 1, 'test': 1, 'hyphenatedword': 2, 'has': 1, 'words': 1}
      result = self.wc.split_text(text)
      self.assertEqual(result, expected)

    def test_sort_normalize_basic(self):
        input_words = {'word1': 3, 'word2': 1, 'word3': 2}
        expected = [('word1', 1.0, 3), ('word3', 2/3, 2), ('word2', 1/3, 1)]
        result = self.wc.sort_normalize(input_words)
        self.assertEqual(result, expected)

    def test_sort_normalize_same_frequency(self):
        input_words = {'word1': 2, 'word2': 2, 'word3': 2}
        expected = [('word1', 1.0, 2), ('word2', 1.0, 2), ('word3', 1.0, 2)]  # Order might vary in Python < 3.7
        result = self.wc.sort_normalize(input_words)
        self.assertEqual(len(result), len(expected))
        for i in range(len(result)):
            self.assertEqual(result[i][0], expected[i][0]) #Check words order
            self.assertEqual(result[i][1], expected[i][1]) #Check normalization
            self.assertEqual(result[i][2], expected[i][2]) #Check frequency

    def test_sort_normalize_one_word(self):
        input_words = {'word1': 5}
        expected = [('word1', 1.0, 5)]
        result = self.wc.sort_normalize(input_words)
        self.assertEqual(result, expected)

    def test_sort_normalize_empty_input(self):
        with self.assertRaises(ValueError):
            self.wc.sort_normalize({})

    def test_sort_normalize_floats(self):
        input_words = {'word1': 3.0, 'word2': 1.5, 'word3': 2.2}
        expected = [('word1', 1.0, 3.0), ('word3', 2.2/3.0, 2.2), ('word2', 1.5/3.0, 1.5)]
        result = self.wc.sort_normalize(input_words)
        self.assertEqual(result, expected)

    def test_sort_normalize_large_numbers(self):
      input_words = {'word1': 1000000, 'word2': 500000, 'word3': 2000000}
      expected = [('word3', 1.0, 2000000), ('word1', 0.5, 1000000), ('word2', 0.25, 500000)]
      result = self.wc.sort_normalize(input_words)
      self.assertEqual(result, expected)

    def test_sort_normalize_mixed_types(self):
        input_words = {'word1': 3, 'word2': 1.5, 'word3': 2}
        expected = [('word1', 1.0, 3), ('word3', 2/3, 2), ('word2', 1.5/3, 1.5)]
        result = self.wc.sort_normalize(input_words)
        self.assertEqual(result, expected)

    def test_sort_normalize_unicode(self):
        input_words = {'wörd1': 3, 'word2': 1, 'word3': 2}
        expected = [('wörd1', 1.0, 3), ('word3', 2/3, 2), ('word2', 1/3, 1)]
        result = self.wc.sort_normalize(input_words)
        self.assertEqual(result, expected)

    def test_sort_normalize_with_zero(self):
        input_words = {'word1': 3, 'word2': 0, 'word3': 2}
        expected = [('word1', 1.0, 3), ('word3', 2/3, 2), ('word2', 0.0, 0)]
        result = self.wc.sort_normalize(input_words)
        self.assertEqual(result, expected)
        
    def test_prepare_text(self):
        text = "This is a test string. This string has some repeated words."
        result = self.wc.prepare_text(text)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result) > 0)

    @patch("wordcloud.Wordcloud.split_text")
    @patch("wordcloud.Wordcloud.sort_normalize")
    def test_prepare_text_basic(self, mock_sort_normalize, mock_split_text):
        text = "This is a test string."
        expected_split = {'this': 1, 'is': 1, 'a': 1, 'test': 1, 'string': 1}
        expected_normalized = [('this', 1.0, 1), ('is', 1.0, 1), ('a', 1.0, 1), ('test', 1.0, 1), ('string', 1.0, 1)]  # Example normalized data
        mock_split_text.return_value = expected_split
        mock_sort_normalize.return_value = expected_normalized

        result = self.wc.prepare_text(text)

        mock_split_text.assert_called_once_with(text, None, None)  # Check if split_text is called
        mock_sort_normalize.assert_called_once_with(expected_split)  # Check if sort_normalize is called
        self.assertEqual(result, expected_normalized)  # Check if the returned value is correct

    @patch("wordcloud.Wordcloud.split_text")
    @patch("wordcloud.Wordcloud.sort_normalize")
    def test_prepare_text_with_stopwords(self, mock_sort_normalize, mock_split_text):
        text = "This isa test string aloha."
        stopwords = ["isa", "aloha"]
        expected_split = {'this': 1, 'test': 1, 'string': 1} #After removing stopwords
        expected_normalized = [('this', 1.0, 1), ('test', 1.0, 1), ('string', 1.0, 1)]  # Example normalized data
        mock_split_text.return_value = expected_split
        mock_sort_normalize.return_value = expected_normalized

        result = self.wc.prepare_text(text, stopwords=stopwords)

        mock_split_text.assert_called_once_with(text, stopwords, None)
        mock_sort_normalize.assert_called_once_with(expected_split)
        self.assertEqual(result, expected_normalized)

    @patch("wordcloud.Wordcloud.split_text")
    @patch("wordcloud.Wordcloud.sort_normalize")
    def test_prepare_text_with_min_word_length(self, mock_sort_normalize, mock_split_text):
        text = "This isa test string."
        min_word_length = 5
        expected_split = {'string': 1} #After applying min_word_length
        expected_normalized = [('string', 1.0, 1)]  # Example normalized data
        mock_split_text.return_value = expected_split
        mock_sort_normalize.return_value = expected_normalized

        result = self.wc.prepare_text(text, min_word_length=min_word_length)

        mock_split_text.assert_called_once_with(text, None, min_word_length)
        mock_sort_normalize.assert_called_once_with(expected_split)
        self.assertEqual(result, expected_normalized)

    @patch("wordcloud.Wordcloud.split_text")
    @patch("wordcloud.Wordcloud.sort_normalize")
    def test_prepare_text_with_stopwords_and_min_word_length(self, mock_sort_normalize, mock_split_text):
        text = "This isa test string. Another longer word. Aloha."
        stopwords = ["isa", "aloha"]
        min_word_length = 5
        expected_split = {'test': 1, 'string': 1, 'another': 1, 'longer': 1, 'word': 1} #After applying both
        expected_normalized = [('test', 1.0, 1), ('string', 1.0, 1), ('another', 1.0, 1), ('longer', 1.0, 1), ('word', 1.0, 1)]  # Example normalized data
        mock_split_text.return_value = expected_split
        mock_sort_normalize.return_value = expected_normalized

        result = self.wc.prepare_text(text, stopwords=stopwords, min_word_length=min_word_length)

        mock_split_text.assert_called_once_with(text, stopwords, min_word_length)
        mock_sort_normalize.assert_called_once_with(expected_split)
        self.assertEqual(result, expected_normalized)

    @patch("wordcloud.Wordcloud.split_text")
    @patch("wordcloud.Wordcloud.sort_normalize")
    def test_prepare_text_empty_string(self, mock_sort_normalize, mock_split_text):
        text = ""
        expected_split = {}
        expected_normalized = []
        mock_split_text.return_value = expected_split
        mock_sort_normalize.return_value = expected_normalized

        result = self.wc.prepare_text(text)

        mock_split_text.assert_called_once_with(text, None, None)
        mock_sort_normalize.assert_called_once_with(expected_split)
        self.assertEqual(result, expected_normalized)

    @patch("wordcloud.IntegralImage")  # Mock the IntegralImage class
    def test_find_position(self, MockIntegralImage):
        frequencies = [("test", 0.5, 2)]
        
        assert MockIntegralImage.find_position(frequencies)
        assert MockIntegralImage.find_position.called
    
        MockIntegralImage.find_position.assert_called_once_with(frequencies)
        
        self.wc.find_position(frequencies)
        self.assertTrue(self.wc.gen_positions)

    @patch("wordcloud.IntegralImage")
    def test_find_position_basic(self, MockIntegralImage):
        frequencies = [("test", 0.5, 2)]
        
        MockIntegralImage.find_position(frequencies)  # Simulate a successful position finding
        MockIntegralImage.find_position.assert_called_once_with(frequencies)
        
        self.wc.find_position(frequencies)
        self.assertTrue(self.wc.gen_positions)
        
        word_data = self.wc.gen_positions[0][0]
        self.assertEqual(word_data[0], "test") # Check if the word is correct
        self.assertEqual([word_data], frequencies)

    @patch("wordcloud.IntegralImage")
    def test_find_position_no_position(self, MockIntegralImage):
        MockIntegralImage.find_position = MagicMock(return_value=None)
        frequencies = [("test", 0.5, 20)]
        
        self.wc_small.find_position(frequencies)
        
        self.assertFalse(self.wc_small.gen_positions)
        self.assertFalse(MockIntegralImage.find_position(frequencies))

    @patch("wordcloud.IntegralImage")
    def test_find_position_multiple_words(self, MockIntegralImage):
        mock_integral_image = MockIntegralImage.return_value
        frequencies = [("test1", 0.5, 2), ("test2", 0.3, 1)]
        self.wc.place_strategy = "brute" # need to change default (random) strategy
        first_word = (0, 0)
        second_word = (60, 0)
        
        mock_integral_image.find_position.side_effect = [first_word, second_word]  # Simulate positions for two words
        
        self.wc.find_position(frequencies)
        
        self.assertEqual(len(self.wc.gen_positions), len(frequencies))
        self.assertEqual(self.wc.gen_positions[0][0][0], "test1") # Check if the word is correct
        self.assertEqual(self.wc.gen_positions[0][3], (first_word[0] + self.wc.margin // 2, first_word[1] + self.wc.margin // 2)) # Check if the position is correct
        self.assertEqual(self.wc.gen_positions[0][0], frequencies[0])
        
        self.assertEqual(self.wc.gen_positions[1][0][0], "test2") # Check if the word is correct
        self.assertEqual(self.wc.gen_positions[1][3], (second_word[0] + self.wc.margin // 2, second_word[1] + self.wc.margin // 2)) # Check if the position is correct
        self.assertEqual(self.wc.gen_positions[1][0], frequencies[1])

    @patch("wordcloud.IntegralImage")
    def test_find_position_wc_too_small(self, MockIntegralImage):
        mock_integral_image = MockIntegralImage.return_value
        mock_integral_image.find_position.return_value = []
        self.wc.min_font_size = 20  # Set min_font_size higher
        frequencies = [("test", 0.5, 2)]
        
        self.wc_small.find_position(frequencies)
        
        self.assertFalse(self.wc_small.gen_positions)  # No positions should be generated
        self.assertEqual(self.wc_small.gen_positions, mock_integral_image.find_position(frequencies))

    def test_find_position_zero_frequency(self):
        frequencies = [("test", 0.0, 2)]  # Word with zero frequency
        self.wc.find_position(frequencies)
        self.assertFalse(self.wc.gen_positions)

    @patch("wordcloud.IntegralImage")
    def test_find_position_rect_only(self, MockIntegralImage):
        mock_integral_image = MockIntegralImage.return_value
        mock_integral_image.find_position.return_value = (79, 96)
        self.wc.place_strategy = "KDTree"
        self.wc.rect_only = True
        frequencies = [("test", 0.5, 2)]
        
        self.wc.find_position(frequencies)
        
        self.assertTrue(self.wc.gen_positions)
        self.assertEqual(self.wc.gen_positions[0][3], mock_integral_image.find_position(frequencies))

    @patch("wordcloud.IntegralImage")
    def test_find_position_custom_max_font_size(self, MockIntegralImage):
        mock_integral_image = MockIntegralImage.return_value
        mock_integral_image.find_position.return_value = (179, 141)
        self.wc.max_font_size = 50
        self.wc.place_strategy = "KDTree"
        frequencies = [("test", 0.5, 2)]
        
        self.wc.find_position(frequencies)
        
        self.assertTrue(self.wc.gen_positions)
        self.assertEqual(self.wc.gen_positions[0][3], mock_integral_image.find_position(frequencies))

    @patch("wordcloud.IntegralImage")
    def test_find_position_single_word_adjust_max_font_size(self, MockIntegralImage):
        mock_integral_image = MockIntegralImage.return_value
        mock_integral_image.find_position.return_value = (2, 62)
        frequencies = [("test", 1.0, 2)]
        self.wc.place_strategy = "rectangular"
        
        self.wc.find_position(frequencies)
        
        self.assertTrue(self.wc.gen_positions)
        self.assertEqual(self.wc.max_font_size, self.wc.height) # Max font size should be adjusted to image height
        self.assertEqual(self.wc.gen_positions[0][3], mock_integral_image.find_position(frequencies))


    
    def test_draw_image_save_file(self, mock_create_folder, mock_save):
        self.wc.generate("Text")
        image = self.wc.draw_image(save_file=True, image_name="test_wordcloud")
        
        self.assertIsInstance(image, Image.Image)
        mock_create_folder.assert_called_once() #Check if folder creation is called
        mock_save.assert_called_once_with(f"{self.wc.results_folder}/test_wordcloud.png", optimize=True)

    @patch("wordcloud.IntegralImage")
    def test_update_position(self, MockIntegralImage):
        updated_pos = [(("test", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red")]
        mock_integral_image = MockIntegralImage.return_value
        mock_integral_image.gen_positions = updated_pos
        self.wc.gen_positions = updated_pos
        new_fonts = {"test": "fonts/Cabal.ttf"} # Simulate new fonts
        
        self.wc.update_position(new_fonts)
        mock_integral_image.update_position(new_fonts)
        
        mock_integral_image.update_position.assert_called_once_with(new_fonts)
        self.assertTrue(self.wc.gen_positions)
        self.assertEqual(self.wc.gen_positions[0][1], "fonts/Cabal.ttf")

    def test_update_position_basic(self):
        first_word = (166, 138)
        second_word = (179, 160)
        third_word = (179, 124)
        
        self.wc.gen_positions = self.initial_positions # Create a copy
        new_fonts = {"test2": FONT_PATH}  # Use default font
        self.wc.place_strategy = "KDTree"
        self.wc.update_position(new_fonts)

        self.assertEqual(len(self.wc.gen_positions), 3)
        self.assertEqual(self.wc.gen_positions[1][1], FONT_PATH)
        self.assertEqual(self.wc.gen_positions[0][3], (first_word[0] + self.wc.margin // 2, first_word[1] + self.wc.margin // 2))
        self.assertEqual(self.wc.gen_positions[1][3], (second_word[0] + self.wc.margin // 2, second_word[1] + self.wc.margin // 2))
        self.assertEqual(self.wc.gen_positions[2][3], (third_word[0] + self.wc.margin // 2, third_word[1] + self.wc.margin // 2))

    def test_update_position_custom_fonts(self):
        self.wc.gen_positions = self.initial_positions
        new_fonts = {"test1": "fonts/Nasa21.ttf"}
        self.wc.update_position(new_fonts)

        self.assertEqual(self.wc.gen_positions[0][1], "fonts/Nasa21.ttf")  # Check if font updated
        self.assertEqual(self.wc.gen_positions[2][1], FONT_PATH)  # Check if other font is default

    def test_update_position_no_position(self):
        self.wc.gen_positions = self.initial_positions
        self.wc.place_strategy = "KDTree"
        new_fonts = {}
        self.wc.update_position(new_fonts)

        self.assertEqual(self.wc.gen_positions, self.update_KDTree_positions)  # No positions should be generated

    def test_update_position_font_too_small(self):
        self.wc.min_font_size = 40  # Set min_font_size higher
        self.wc.gen_positions = self.initial_positions
        new_fonts = {}
        self.wc.update_position(new_fonts)
        
        self.assertEqual(self.wc.gen_positions, [])

    def test_update_position_multiple_words(self):
        self.wc.gen_positions = self.initial_positions
        self.wc.place_strategy = "KDTree"
        new_fonts = {}
        self.wc.update_position(new_fonts)

        self.assertEqual(len(self.wc.gen_positions), 3)
        self.assertEqual(self.wc.gen_positions[0][3], (166 + self.wc.margin // 2, 138 + self.wc.margin // 2))  # Check updated position
        self.assertEqual(self.wc.gen_positions[1][3], (179 + self.wc.margin // 2, 160 + self.wc.margin // 2))  # Check updated position
        self.assertEqual(self.wc.gen_positions[2][3], (179 + self.wc.margin // 2, 124 + self.wc.margin // 2))
        
        self.assertEqual(self.wc.gen_positions[0][1], FONT_PATH)
        self.assertEqual(self.wc.gen_positions[1][1], FONT_PATH)
        self.assertEqual(self.wc.gen_positions[2][1], FONT_PATH)

    def test_update_position_empty_gen_positions(self):
        self.wc.gen_positions = []  # Empty
        new_fonts = {}
        self.wc.update_position(new_fonts)
        self.assertEqual(self.wc.gen_positions, [])  # Should remain empty

    def test_update_position_single_word_adjust_max_font_size(self):
        self.wc.gen_positions = [(("test", 1.0, 2), "fonts/Nasa21.ttf", 20, (50, 50), None, "red")] # Only one word
        new_fonts = {}
        self.wc.update_position(new_fonts)
        self.assertEqual(self.wc.max_font_size, self.wc.height) # Max font size should be adjusted to image height
        self.assertEqual(self.wc.gen_positions[0][1], FONT_PATH)

    def test_update_colors(self):
        self.wc.gen_positions = self.initial_positions
        new_colors = {"test1": "green"}
        self.wc.update_colors(new_colors)
        self.assertEqual(self.wc.gen_positions[0][5], "green")  # Check if color updated
        self.assertEqual(self.wc.gen_positions[1][5], "blue")  # Check if other color unchanged
        self.assertEqual(self.wc.gen_positions[2][5], "green")
        
    def test_update_colors_two(self):
        new_colors = {"test1": "purple", "test2": "orange"}
        self.wc.gen_positions = self.initial_positions
        self.wc.update_colors(new_colors)

        self.assertEqual(len(self.wc.gen_positions), 3)
        self.assertEqual(self.wc.gen_positions[0][5], "purple")  # Check updated color
        self.assertEqual(self.wc.gen_positions[1][5], "orange")  # Check updated color
        self.assertEqual(self.wc.gen_positions[2][5], "green")  # Check if other color unchanged

    def test_update_colors_empty(self):
        new_colors = {}  # Empty dictionary
        self.wc.gen_positions = self.initial_positions
        self.wc.update_colors(new_colors)

        self.assertEqual(self.wc.gen_positions[0][5], "red")  # Check if colors remain the same
        self.assertEqual(self.wc.gen_positions[1][5], "blue")
        self.assertEqual(self.wc.gen_positions[2][5], "green")

    def test_update_colors_no_match(self):
        new_colors = {"test4": "yellow"}  # No matching word
        self.wc.gen_positions = self.initial_positions
        self.wc.update_colors(new_colors)

        self.assertEqual(self.wc.gen_positions[0][5], "red")  # Check if colors remain the same
        self.assertEqual(self.wc.gen_positions[1][5], "blue")
        self.assertEqual(self.wc.gen_positions[2][5], "green")

    def test_update_colors_empty_gen_positions(self):
        self.wc.gen_positions = []  # Empty gen_positions
        new_colors = {"test1": "purple"}
        self.wc.update_colors(new_colors)
        self.assertEqual(self.wc.gen_positions, []) #Should remain empty

    def test_update_colors_with_unicode(self):
        self.wc.gen_positions = [
            (("tést1", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red"), #unicode in word
        ]
        new_colors = {"tést1": "purple"}
        self.wc.update_colors(new_colors)
        self.assertEqual(self.wc.gen_positions[0][5], "purple")

    def test_update_colors_with_rgb(self):
        self.wc.gen_positions = [
            (("test1", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red"), #unicode in word
        ]
        new_colors = {"test1": "rgb(0, 0, 0)"}
        self.wc.update_colors(new_colors)
        self.assertEqual(self.wc.gen_positions[0][5], "rgb(0, 0, 0)")

    def test_update_colors_with_rgb_partialy(self):
        self.wc.gen_positions = self.initial_positions
        new_colors = {"test1": "rgb(1, 1, 1)", "test2": "rgb(2, 2, 2)"}
        self.wc.update_colors(new_colors)
        
        self.assertEqual(self.wc.gen_positions[0][5], "rgb(1, 1, 1)")
        self.assertEqual(self.wc.gen_positions[1][5], "rgb(2, 2, 2)")
        self.assertEqual(self.wc.gen_positions[2][5], "green")
        
    def test_update_colors_with_non_rgb(self):
        self.wc.gen_positions = [
            (("test1", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red"), #unicode in word
        ]
        new_colors = {"test1": "rgb(300, 300, 300)"}
        self.wc.update_colors(new_colors)
        self.assertEqual(self.wc.gen_positions[0][5], "rgb(300, 300, 300)")

    def test_generate(self):
        text = "This is a test string."
        self.wc.generate(text)
        self.assertTrue(self.wc.gen_positions is not None)
        
    @patch("wordcloud.Wordcloud.prepare_text")
    @patch("wordcloud.Wordcloud.find_position")
    def test_generate_basic(self, mock_find_position, mock_prepare_text):
        text = "This is a test string."
        expected_prepared = [("test", 1.0, 1), ("this", 1.0, 1), ("string", 1.0, 1)]  # Example prepared data
        expected_result = self.wc #Return self
        mock_prepare_text.return_value = expected_prepared
        mock_find_position.return_value = expected_result

        result = self.wc.generate(text)

        mock_prepare_text.assert_called_once_with(text)
        mock_find_position.assert_called_once_with(expected_prepared)
        self.assertEqual(result, expected_result)

    @patch("wordcloud.Wordcloud.prepare_text")
    @patch("wordcloud.Wordcloud.find_position")
    def test_generate_empty_text(self, mock_find_position, mock_prepare_text):
        text = ""
        expected_prepared = []
        expected_result = self.wc #Return self
        mock_prepare_text.return_value = expected_prepared
        mock_find_position.return_value = expected_result

        result = self.wc.generate(text)

        mock_prepare_text.assert_called_once_with(text)
        mock_find_position.assert_called_once_with(expected_prepared)
        self.assertEqual(result, expected_result)

    @patch("wordcloud.Wordcloud.prepare_text")
    @patch("wordcloud.Wordcloud.find_position")
    def test_generate_with_params(self, mock_find_position, mock_prepare_text):
        text = "This is a test string."
        stopword = ["this"]
        min_word_length = 4
        expected_prepared = [("test", 1.0, 1), ("string", 1.0, 1)]  # Example prepared data (after filtering)
        self.wc.stopwords = stopword
        self.wc.min_word_length = min_word_length
        expected_result = self.wc #Return self
        mock_prepare_text.return_value = expected_prepared
        mock_find_position.return_value = expected_result

        result = self.wc.generate(text)

        mock_prepare_text.assert_called_once_with(text)
        mock_find_position.assert_called_once_with(expected_prepared)
        self.assertEqual(result, expected_result)

    @patch("wordcloud.Wordcloud.prepare_text")
    @patch("wordcloud.Wordcloud.find_position")
    def test_generate_returns_self(self, mock_find_position, mock_prepare_text):
        text = "Test"
        expected_prepared = [("test", 1.0, 1)]
        expected_result = self.wc
        mock_prepare_text.return_value = expected_prepared
        mock_find_position.return_value = expected_result
        result = self.wc.generate(text)
        self.assertIs(result, self.wc)

    @patch("wordcloud.Wordcloud.create_folder") #Mock folder creation
    def test_draw_image_basic(self, mock_create_folder):
        self.wc.generate("Text")
        image = self.wc.draw_image()
        
        self.assertIsInstance(image, Image.Image)  # Check if it's a PIL Image object
        self.assertEqual(image.size, (self.wc.width, self.wc.height))  # Check correct dimensions
        mock_create_folder.assert_not_called() #Check if folder creation is not called
    
    def test_draw_image_basic_error(self):
        with self.assertRaises(AttributeError): #Expect error, WC not yet generated
          self.wc.draw_image()

    @patch('PIL.Image.Image.save')
    @patch("wordcloud.Wordcloud.create_folder")
    def test_draw_image_save_file(self, mock_create_folder, mock_save):
        self.wc.generate("Text")
        image = self.wc.draw_image(save_file=True, image_name="test_wordcloud")
        
        self.assertIsInstance(image, Image.Image)
        mock_create_folder.assert_called_once() #Check if folder creation is called
        mock_save.assert_called_once_with(f"{self.wc.results_folder}/test_wordcloud.png", optimize=True)

    def test_draw_image_empty_positions(self):
        self.wc.gen_positions = []  # No positions
        image = self.wc.draw_image()
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (self.wc.width, self.wc.height))

    def test_draw_image_custom_background(self):
        self.wc.background_color = "yellow"
        self.wc.gen_positions = self.initial_positions
        image = self.wc.draw_image()
        
        self.assertIsInstance(image, Image.Image)
        # Add a more robust check for background color if needed (e.g., using image pixel data)

    def test_draw_image_custom_mode(self):
        self.wc.mode = "RGBA"  # Test with RGBA mode
        self.wc.gen_positions = self.initial_positions
        
        image = self.wc.draw_image()
        
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGBA")

    def test_draw_image_black_white(self):
      self.wc.black_white = True
      self.wc.gen_positions = self.initial_positions
      image = self.wc.draw_image()
      self.assertIsInstance(image, Image.Image)

    @patch("wordcloud.Wordcloud.draw_image")  # Mock draw_image
    @patch("matplotlib.pyplot.figure")  # Mock plt.figure
    @patch("matplotlib.pyplot.imshow")  # Mock plt.imshow
    @patch("matplotlib.pyplot.axis")  # Mock plt.axis
    @patch("matplotlib.pyplot.show")  # Mock plt.show
    def test_draw_plt_image_basic(self, mock_show, mock_axis, mock_imshow, mock_figure, mock_draw_image):
        mock_image = MagicMock()  # Mock PIL Image object
        mock_draw_image.return_value = mock_image

        mock_figure_instance = mock_figure.return_value #Mock figure instance
        result = self.wc.draw_plt_image()

        mock_draw_image.assert_called_once()
        mock_figure.assert_called_once()
        mock_imshow.assert_called_once_with(mock_image, interpolation="bilinear")
        mock_axis.assert_called_once_with("off")
        mock_show.assert_called_once()
        self.assertIs(result, self.wc) #Check return self

    @patch("wordcloud.Wordcloud.draw_image")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.axis")
    @patch("matplotlib.pyplot.show")
    def test_draw_plt_image_empty_positions(self, mock_show, mock_axis, mock_imshow, mock_figure, mock_draw_image):
        self.wc.gen_positions = []
        mock_image = MagicMock()
        mock_draw_image.return_value = mock_image

        mock_figure_instance = mock_figure.return_value #Mock figure instance
        result = self.wc.draw_plt_image()

        mock_draw_image.assert_called_once()
        mock_figure.assert_called_once()
        mock_imshow.assert_called_once_with(mock_image, interpolation="bilinear")
        mock_axis.assert_called_once_with("off")
        mock_show.assert_called_once()
        self.assertIs(result, self.wc) #Check return self

    def test_generate_svg(self):
        self.wc.gen_positions = [
            (("test", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red")
        ]

        svg_content = self.wc.generate_svg()
        self.assertIsInstance(svg_content, str)
        self.assertTrue("<svg" in svg_content)
        
    @patch("builtins.open", new_callable=mock_open)  # Mock file open
    @patch("wordcloud.Wordcloud.create_folder") #Mock folder creation
    def test_generate_svg_save_file(self, mock_create_folder, mock_file):
        self.wc.gen_positions = self.initial_positions
        svg_content = self.wc.generate_svg(save_file=True, file_name="test_wordcloud")
        mock_create_folder.assert_called_once() #Check if folder creation is called
        mock_file.assert_called_once_with(f"{self.wc.results_folder}/test_wordcloud.svg", "w", encoding="utf-8")
        self.assertIsInstance(svg_content, str)
        self.assertTrue("<svg" in svg_content)  # Basic check for SVG tag

    def test_generate_svg_no_save(self):
        self.wc.gen_positions = self.initial_positions
        svg_content = self.wc.generate_svg()
        self.assertIsInstance(svg_content, str)
        self.assertTrue("<svg" in svg_content)  # Basic check for SVG tag

    def test_generate_svg_empty_positions(self):
        self.wc.gen_positions = []
        svg_content = self.wc.generate_svg()
        self.assertIsInstance(svg_content, str)
        self.assertTrue("<svg" in svg_content)  # Basic check for SVG tag

    def test_generate_svg_custom_background(self):
        self.wc.gen_positions = self.initial_positions
        self.wc.background_color = "yellow"
        svg_content = self.wc.generate_svg()
        self.assertIsInstance(svg_content, str)
        self.assertTrue("fill:yellow" in svg_content)  # Check for background style

    def test_generate_svg_no_font_path(self):
        self.wc.gen_positions = self.initial_positions
        self.wc.font_path = None
        with self.assertRaises(AttributeError): #Expect error
          self.wc.generate_svg()

    def test_generate_svg_with_unicode(self):
        self.wc.gen_positions = [
            (("tést1", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red"), #unicode in word
        ]
        svg_content = self.wc.generate_svg()
        self.assertIsInstance(svg_content, str)
        self.assertTrue("tést1" in svg_content)

    @patch("builtins.open", new_callable=mock_open)
    @patch("wordcloud.Wordcloud.create_folder")
    def test_generate_svg_os_error(self, mock_create_folder, mock_file):
        self.wc.gen_positions = self.initial_positions
        mock_file.side_effect = OSError("Simulated OSError")
        svg_content = self.wc.generate_svg(save_file=True)
        mock_create_folder.assert_called_once()
        self.assertIsInstance(svg_content, str)
        self.assertTrue("<svg" in svg_content)  # Basic check for SVG tag

    @patch("builtins.open", mock_open=MagicMock())
    @patch("wordcloud.Wordcloud.create_folder")
    def test_create_html(self, mock_create_folder, mock_file):
        svg_content = "<svg></svg>"
        
        self.wc.create_html(svg_content, save_file=True)
        mock_create_folder.assert_called_once()
        mock_file.assert_called_once_with(f"{self.wc.results_folder}/wordcloud.html", "a")

        html_content = self.wc.create_html(svg_content)
        self.assertIsInstance(html_content, str)
        self.assertTrue("<html" in html_content)
        
    @patch("builtins.open", new_callable=mock_open)
    @patch("wordcloud.Wordcloud.create_folder")
    def test_create_html_save_file(self, mock_create_folder, mock_file):
        svg_content = "<svg>test svg</svg>"
        html_content = self.wc.create_html(svg_content, save_file=True, file_name="test_wordcloud")
        mock_create_folder.assert_called_once()
        mock_file.assert_called_once_with(f"{self.wc.results_folder}/test_wordcloud.html", "a")
        self.assertIsInstance(html_content, str)
        self.assertTrue("<html" in html_content)  # Basic check for HTML tag
        self.assertTrue(svg_content in html_content)

    def test_create_html_no_save(self):
        svg_content = "<svg>test svg</svg>"
        html_content = self.wc.create_html(svg_content)
        self.assertIsInstance(html_content, str)
        self.assertTrue("<html" in html_content)
        self.assertTrue(svg_content in html_content)

    def test_create_html_empty_svg(self):
        svg_content = ""
        html_content = self.wc.create_html(svg_content)
        self.assertIsInstance(html_content, str)
        self.assertTrue("<html" in html_content)

    @patch("os.makedirs")
    def test_create_folder(self, mock_makedirs):
        folder_name = "test_folder"
        self.wc.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)
        
    @patch("os.makedirs")
    def test_create_folder_existing(self, mock_makedirs):
        folder_name = "test_folder"
        mock_makedirs.side_effect = FileExistsError  # Simulate existing folder
        self.wc.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)  # Still called with exist_ok=True

    @patch("os.makedirs")
    def test_create_folder_os_error(self, mock_makedirs):
        folder_name = "test_folder"
        mock_makedirs.side_effect = OSError("Simulated OSError")  # Simulate an OSError
        self.wc.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)

    @patch("os.makedirs")
    def test_create_folder_nested(self, mock_makedirs):
        folder_name = "path/to/folder"
        self.wc.create_folder(folder_name)
        mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)

    @patch('PIL.Image.Image.save')
    @patch("wordcloud.Wordcloud.create_folder")
    def test_draw_image(self, mock_create_folder, mock_save):
        self.wc.gen_positions = [
            (("test", 0.5, 2), FONT_PATH, 20, (50, 50), None, "red")
        ]
        self.wc.draw_image(save_file=True)
        mock_create_folder.assert_called_once()
        mock_save.assert_called_once()

        image = self.wc.draw_image()
        self.assertIsInstance(image, Image.Image)
        
    def test_stopwords_case_sensitive(self):
        self.wc.stopwords = ["the"]
        text = "The quick brown fox jumps over the lazy dog. thE" #stopwords check is case sensitive
        expected = {'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1, 'the': 2}
        result = self.wc.split_text(text)
        self.assertEqual(result, expected)

    def test_min_word_length_zero(self):
        self.wc.min_word_length = 0
        text = "a ab abc abcd"
        expected = {'a': 1, 'ab': 1, 'abc': 1, 'abcd': 1}
        result = self.wc.split_text(text) #Test split_text directly
        self.assertEqual(result, expected)

    #TODO invalid font check
    # def test_invalid_font_path(self):
    #     with self.assertRaises(OSError):  # Or a custom exception if you handle it differently
    #         Wordcloud(font_path="invalid_font_path.ttf")  # Test during initialization

    #TODO invalid color check
    # def test_invalid_color_input(self):
    #     self.wc.color_map = "invalid_colormap"
    #     with self.assertRaises(ValueError):
    #         self.wc.generate("test")

    def test_small_image_size(self):
        wc = Wordcloud(width=50, height=50, font_path=FONT_PATH)
        wc.generate("test")
        image = wc.draw_image()
        self.assertEqual(image.size, (50, 50))

    #TODO background color check
    # def test_transparent_background(self):
    #     self.wc.background_color = None  # or (255, 255, 255, 0) for RGBA
    #     image = self.wc.draw_image()
    #     # Add assertion to check for transparency (requires access to pixel data)

    def test_different_mode(self):
        self.wc.mode = "RGBA"
        self.wc.gen_positions = self.initial_positions
        image = self.wc.draw_image()
        self.assertEqual(image.mode, "RGBA")

    #TODO escaping
    # def test_svg_special_chars(self):
    #     text = "<test & > word\"'"
    #     self.wc.generate(text)
    #     svg = self.wc.generate_svg()
    #     self.assertTrue("&lt;" in svg)  # Check for proper escaping

    #TODO invalid input check
    # def test_generate_invalid_input(self):
    #     with self.assertRaises(TypeError):
    #         self.wc.generate(123)  # Non-string input

    def test_generate_long_text(self):
        with open("./test_sources/generated.txt", "r", encoding="utf-8") as f:
            long_text = f.read()
        self.wc.generate(long_text)
        image = self.wc.draw_image()
        self.assertIsInstance(image, Image.Image)

    def test_performance_long_text(self):
        with open("./test_sources/generated.txt", "r", encoding="utf-8") as f:  # Replace with path
            long_text = f.read()

        import time
        start_time = time.time()
        self.wc.generate(long_text)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds") #Check time manually

        self.assertLess(elapsed_time, 5)  # Set a reasonable threshold (adjust as needed)

if __name__ == "__main__":
    unittest.main()