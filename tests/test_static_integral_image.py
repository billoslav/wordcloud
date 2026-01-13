import unittest

import numpy as np

from wordcloud.utils.integral_image import StaticIntegralImage


class TestStaticIntegralImage(unittest.TestCase):
    def setUp(self):
        self.height = 50
        self.width = 60
        mask_bool = np.zeros((self.height, self.width), dtype=bool)
        mask_bool[10:20, 10:20] = True
        self.sii_mask = np.where(mask_bool, 0, 1).astype(np.uint8)

    def test_initialization_no_mask(self):
        sii = StaticIntegralImage(self.height, self.width)
        self.assertEqual(sii.height, self.height)
        self.assertEqual(sii.width, self.width)
        self.assertEqual(sii.integral[-1, -1], 0)

    def test_initialization_with_mask(self):
        sii = StaticIntegralImage(self.height, self.width, mask_array=self.sii_mask)
        self.assertEqual(sii.integral[-1, -1], 100)

    def test_initialization_invalid_mask_shape(self):
        invalid_mask = np.zeros((self.height // 2, self.width // 2))
        with self.assertRaisesRegex(ValueError, "Mask array shape .* does not match"):
            StaticIntegralImage(self.height, self.width, mask_array=invalid_mask)

    def test_is_valid_position_no_mask(self):
        sii = StaticIntegralImage(self.height, self.width, mask_array=None)
        self.assertTrue(sii.is_valid_position(0, 0, 10, 10))
        self.assertTrue(sii.is_valid_position(self.height - 10, 0, 10, 10))
        self.assertTrue(sii.is_valid_position(0, self.width - 10, 10, 10))
        self.assertTrue(sii.is_valid_position(self.height - 10, self.width - 10, 10, 10))
        self.assertFalse(sii.is_valid_position(self.height - 5, 0, 10, 10))
        self.assertFalse(sii.is_valid_position(0, self.width - 5, 10, 10))
        self.assertFalse(sii.is_valid_position(self.height, 0, 10, 10))
        self.assertFalse(sii.is_valid_position(0, self.width, 10, 10))
        self.assertFalse(sii.is_valid_position(-1, 0, 10, 10))
        self.assertFalse(sii.is_valid_position(0, -1, 10, 10))
        self.assertFalse(sii.is_valid_position(0, 0, -10, 10))
        self.assertFalse(sii.is_valid_position(0, 0, 10, -10))
        self.assertFalse(sii.is_valid_position(0, 0, 0, 10))
        self.assertFalse(sii.is_valid_position(0, 0, 10, 0))

    def test_is_valid_position_with_mask(self):
        sii = StaticIntegralImage(self.height, self.width, mask_array=self.sii_mask)
        self.assertTrue(sii.is_valid_position(0, 0, 5, 5))
        self.assertTrue(sii.is_valid_position(20, 20, 10, 10))
        self.assertTrue(sii.is_valid_position(5, 25, 10, 10))
        self.assertFalse(sii.is_valid_position(5, 5, 10, 10))
        self.assertFalse(sii.is_valid_position(10, 10, 5, 5))
        self.assertFalse(sii.is_valid_position(15, 15, 10, 10))
        self.assertFalse(sii.is_valid_position(5, 15, 10, 10))
        self.assertFalse(sii.is_valid_position(15, 5, 10, 10))
        self.assertFalse(sii.is_valid_position(45, 55, 10, 10))


if __name__ == "__main__":
    unittest.main()

