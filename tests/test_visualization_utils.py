import unittest

from wordcloud.utils import (
    COLOR_THEMES,
    generate_color_gradient,
    generate_colors_by_frequency,
    generate_colors_by_sentiment,
    generate_colors_by_length,
)


class TestVisualizationUtils(unittest.TestCase):
    def test_generate_color_gradient(self):
        start = "#000000"
        end = "#ffffff"
        colors = generate_color_gradient(start, end, 3)
        self.assertEqual(colors[0], "#000000")
        self.assertEqual(colors[-1], "#ffffff")
        self.assertEqual(len(colors), 3)

    def test_generate_colors_by_frequency(self):
        freqs = {"a": 3, "b": 2, "c": 1}
        colors = generate_colors_by_frequency(freqs, color_theme="default", random_colors=False, shuffle=False)
        self.assertEqual(set(colors.keys()), set(freqs.keys()))
        # Highest frequency gets first palette color
        self.assertEqual(colors["a"], COLOR_THEMES["default"][0])

    def test_generate_colors_by_sentiment(self):
        words = ["pos", "neg", "neu"]
        sentiments = {"pos": 0.5, "neg": -0.5}
        colors = generate_colors_by_sentiment(words, sentiments)
        self.assertNotEqual(colors["pos"], colors["neg"])
        self.assertNotEqual(colors["neu"], colors["pos"])
        self.assertNotEqual(colors["neu"], colors["neg"])

    def test_generate_colors_by_length(self):
        words = ["aa", "aaaaaa", "aaaaaaaaaaaa"]
        colors = generate_colors_by_length(words, min_length=2, max_length=12)
        self.assertEqual(set(colors.keys()), set(words))
        for color in colors.values():
            self.assertTrue(color.startswith("#"))


if __name__ == "__main__":
    unittest.main()

