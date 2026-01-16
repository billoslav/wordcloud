"""
Tests for custom placement and collision plugin registration.
"""

from __future__ import annotations

from wordcloud import Wordcloud
from wordcloud.utils.placement import register_placement_strategy, unregister_placement_strategy
from wordcloud.utils.collision import (
    CollisionDetector,
    register_collision_detector,
    unregister_collision_detector,
    create_collision_detector,
)


def test_custom_placement_strategy() -> None:
    def top_left_strategy(integral_image, free_locations, width_x, height_y, size_x, size_y):
        return min(free_locations) if free_locations else None

    register_placement_strategy("custom_test", top_left_strategy)
    try:
        wc = Wordcloud(
            width=200,
            height=100,
            font_path="fonts/Arial Unicode.ttf",
            place_strategy="custom_test",
            random_state=123,
            max_words=20,
        )
        wc.generate("custom strategy test text " * 10)
        assert wc.gen_positions is not None
    finally:
        unregister_placement_strategy("custom_test")


def test_custom_collision_detector() -> None:
    class SimpleCollisionDetector(CollisionDetector):
        def __init__(self, width: int, height: int, **kwargs):
            super().__init__(width, height, rng=kwargs.get("rng"))
            self.rectangles = []

        def add_rectangle(self, rect):
            self.rectangles.append(rect)

        def check_collision(self, rect):
            rect_x, rect_y, rect_w, rect_h = rect
            for r_x, r_y, r_w, r_h in self.rectangles:
                if (rect_x < r_x + r_w and rect_x + rect_w > r_x and
                    rect_y < r_y + r_h and rect_y + rect_h > r_y):
                    return True
            return False

        def clear(self):
            self.rectangles = []

    register_collision_detector("custom_collision", lambda width, height, **kwargs: SimpleCollisionDetector(width, height, **kwargs))
    try:
        detector = create_collision_detector("custom_collision", width=200, height=100)
        wc = Wordcloud(
            width=200,
            height=100,
            font_path="fonts/Arial Unicode.ttf",
            place_strategy="random",
            collision_detector=detector,
            random_state=123,
            max_words=20,
        )
        wc.generate("collision detector test text " * 10)
        assert wc.gen_positions is not None
    finally:
        unregister_collision_detector("custom_collision")
