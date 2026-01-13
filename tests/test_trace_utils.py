import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from wordcloud.utils.trace_utils import (
    TRACE_MARGIN,
    Tracer,
    create_tracking_structure,
    draw_trace_point,
    save_trace_img,
)

TEST_TRACKING_DIR = Path("./TestTracking")


class TestTraceUtils(unittest.TestCase):
    def setUp(self):
        if TEST_TRACKING_DIR.exists():
            shutil.rmtree(TEST_TRACKING_DIR)
        self.base_args = {
            "canvas_width": 100,
            "canvas_height": 80,
            "base_tracking_dir": TEST_TRACKING_DIR,
            "place_strategy": "test_strategy",
            "word_info": ("testword", 1.0, 1),
            "word_size": (20, 10),
            "mask_array": None,
        }
        self.tracer = Tracer()

    def tearDown(self):
        if TEST_TRACKING_DIR.exists():
            shutil.rmtree(TEST_TRACKING_DIR)

    @patch("PIL.Image.new")
    @patch("PIL.ImageDraw.Draw")
    @patch("pathlib.Path.mkdir")
    def test_setup_creates_structure_and_image(
        self, mock_mkdir, mock_draw_constructor, mock_image_constructor
    ):
        mock_img = MagicMock(spec=Image.Image)
        mock_image_constructor.return_value = mock_img

        self.tracer.setup(**self.base_args)

        expected_strategy_dir = TEST_TRACKING_DIR / "test_strategy"
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

        expected_width = self.base_args["canvas_width"] + TRACE_MARGIN
        expected_height = self.base_args["canvas_height"] + TRACE_MARGIN
        mock_image_constructor.assert_called_once_with("RGB", (expected_width, expected_height), color="white")
        mock_draw_constructor.assert_called_once_with(mock_img)
        self.assertIsNotNone(self.tracer.trace_img)
        self.assertIsNotNone(self.tracer.trace_draw)
        expected_prefix = str(expected_strategy_dir / "trace_testword")
        self.assertTrue(str(self.tracer.trace_img_name).startswith(expected_prefix))
        self.assertTrue(self.tracer.trace_img_name.endswith(".png"))

    @patch("PIL.Image.new")
    @patch("PIL.ImageDraw.Draw")
    @patch("pathlib.Path.mkdir")
    def test_setup_with_mask(self, mock_mkdir, mock_draw_constructor, mock_image_constructor):
        mock_img = MagicMock(spec=Image.Image)
        mock_image_constructor.return_value = mock_img
        mask = np.ones((self.base_args["canvas_height"], self.base_args["canvas_width"]), dtype=np.uint8)
        mask[10:20, 10:20] = 0

        mock_mask_img = MagicMock(spec=Image.Image)
        mock_mask_img.size = (self.base_args["canvas_width"], self.base_args["canvas_height"])
        with patch("PIL.Image.fromarray", return_value=mock_mask_img) as mock_fromarray:
            args = {**self.base_args, "mask_array": mask}
            self.tracer.setup(**args)
            mock_fromarray.assert_called_once()
            mock_img.paste.assert_called_once()

    @patch("PIL.ImageDraw.Draw")
    def test_draw_point(self, mock_draw_constructor):
        with patch("PIL.Image.new") as mock_new, patch("pathlib.Path.mkdir"):
            mock_img = MagicMock(spec=Image.Image)
            mock_new.return_value = mock_img
            self.tracer.setup(**self.base_args)

        test_x, test_y = 30, 40
        self.tracer.draw_point(test_x, test_y)

        mock_draw_instance = mock_draw_constructor.return_value
        mock_draw_instance.point.assert_called_once()
        args, kwargs = mock_draw_instance.point.call_args
        expected_x = test_x + TRACE_MARGIN // 2
        expected_y = test_y + TRACE_MARGIN // 2
    
        self.assertEqual(args[0], [(expected_x, expected_y)])
        self.assertEqual(kwargs["fill"], "red")

    @patch("wordcloud.utils.trace_utils.save_trace_img")
    def test_save_calls_utility(self, mock_save_trace_img):
        with patch("PIL.Image.new") as mock_new, patch("PIL.ImageDraw.Draw"), patch("pathlib.Path.mkdir"):
            mock_img = MagicMock(spec=Image.Image)
            mock_new.return_value = mock_img
            self.tracer.setup(**self.base_args)

        mock_img_obj = self.tracer.trace_img
        trace_name_before_reset = self.tracer.trace_img_name

        self.tracer.save()

        mock_save_trace_img.assert_called_once_with(mock_img_obj, trace_name_before_reset)
        self.assertFalse(self.tracer.is_active)

    @patch("wordcloud.utils.trace_utils.save_trace_img")
    def test_save_wrapper(self, mock_save_func):
        self.tracer.is_active = True
        self.tracer.trace_img = MagicMock(spec=Image.Image)
        self.tracer.trace_img_name = Path("some/path/trace.png")

        self.tracer.save()

        mock_save_func.assert_called_once_with(self.tracer.trace_img, self.tracer.trace_img_name)
        self.assertFalse(self.tracer.is_active)

    def test_save_no_setup(self):
        with patch("PIL.Image.Image.save") as mock_save:
            self.tracer.save()
            mock_save.assert_not_called()


class TestHelpers(unittest.TestCase):
    def tearDown(self):
        if TEST_TRACKING_DIR.exists():
            shutil.rmtree(TEST_TRACKING_DIR)

    def test_create_tracking_structure(self):
        if TEST_TRACKING_DIR.exists():
            shutil.rmtree(TEST_TRACKING_DIR)
        strategy_dir = create_tracking_structure(TEST_TRACKING_DIR, "s")
        self.assertTrue(strategy_dir.exists())
        self.assertTrue(strategy_dir.is_dir())

    def test_save_trace_img(self):
        tmp = Path("tmp.png")
        img = MagicMock(spec=Image.Image)
        save_trace_img(img, tmp)
        img.save.assert_called_once_with(tmp)
        if tmp.exists():
            os.remove(tmp)


if __name__ == "__main__":
    unittest.main()

