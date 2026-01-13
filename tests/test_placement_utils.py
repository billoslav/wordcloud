import unittest
from unittest.mock import MagicMock

from wordcloud.utils import placement


class TestPlacementUtils(unittest.TestCase):
    def setUp(self):
        self.width = 100
        self.height = 80
        self.place_w = 10
        self.place_h = 5

    def test_find_position_random(self):
        pos = placement.find_position_random(self.width, self.height, self.place_w, self.place_h, is_valid)
        self.assertIsNotNone(pos)
        self.assertTrue(0 <= pos[0] <= self.width - self.place_w)
        self.assertTrue(0 <= pos[1] <= self.height - self.place_h)

    def test_find_position_random_no_space(self):
        is_valid = MagicMock(return_value=False)
        max_attempts = 50

        pos = placement.find_position_random(
            self.width, self.height, self.place_w, self.place_h, is_valid, max_attempts=max_attempts
        )

        self.assertIsNone(pos)
        self.assertEqual(is_valid.call_count, max_attempts)

    def test_find_position_random_invalid_size(self):
        is_valid = MagicMock(return_value=True)

        pos = placement.find_position_random(
            self.width, self.height, self.width + 1, self.height + 1, is_valid
        )

        self.assertIsNone(pos)
        is_valid.assert_not_called()

    def test_find_position_rectangular_spiral(self):
        is_valid = MagicMock(return_value=True)

        pos = placement.find_position_rectangular_spiral(
            self.width, self.height, self.place_w, self.place_h, is_valid
        )

        expected_x = (self.width - self.place_w) // 2
        expected_y = (self.height - self.place_h) // 2
        is_valid.assert_called_with(expected_y, expected_x, self.place_h, self.place_w)
        self.assertEqual(pos, (expected_x, expected_y))

    def test_find_position_rectangular_spiral_blocked_center(self):
        center_x = (self.width - self.place_w) // 2
        center_y = (self.height - self.place_h) // 2

        def validity_checker(y, x, h, w):
            if x == center_x and y == center_y:
                return False
            return True

        is_valid = MagicMock(side_effect=validity_checker)

        pos = placement.find_position_rectangular_spiral(
            self.width, self.height, self.place_w, self.place_h, is_valid
        )

        self.assertIsNotNone(pos)
        self.assertGreater(is_valid.call_count, 1)
        self.assertNotEqual(pos, (center_x, center_y))

    def test_find_position_rectangular_spiral_no_space(self):
        is_valid = MagicMock(return_value=False)

        pos = placement.find_position_rectangular_spiral(
            self.width, self.height, self.place_w, self.place_h, is_valid
        )

        self.assertIsNone(pos)
        self.assertGreater(is_valid.call_count, 1)

    def test_find_position_rectangular_spiral_invalid_size(self):
        is_valid = MagicMock(return_value=True)

        pos = placement.find_position_rectangular_spiral(
            self.width, self.height, self.width + 1, self.height + 1, is_valid
        )

        self.assertIsNone(pos)
        is_valid.assert_not_called()


def is_valid(pos_y, pos_x, size_y, size_x):
    return True


if __name__ == "__main__":
    unittest.main()

