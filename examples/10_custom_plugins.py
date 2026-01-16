"""
Example: custom placement strategy and collision detector plugin.
"""

from wordcloud import Wordcloud
from wordcloud.utils.placement import register_placement_strategy, unregister_placement_strategy
from wordcloud.utils.collision import CollisionDetector, register_collision_detector, unregister_collision_detector, create_collision_detector


def top_left_strategy(integral_image, free_locations, width_x, height_y, size_x, size_y):
    if not free_locations:
        return None
    return min(free_locations)


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


def main() -> None:
    sample_text = "custom plugin example " * 200
    font_path = "fonts/Arial Unicode.ttf"

    register_placement_strategy("top_left", top_left_strategy)
    register_collision_detector("simple", lambda width, height, **kwargs: SimpleCollisionDetector(width, height, **kwargs))

    try:
        wc = Wordcloud(width=400, height=200, font_path=font_path, place_strategy="top_left")
        wc.generate(sample_text).draw_image(save_file=True, image_name="10_custom_plugins")

        detector = create_collision_detector("simple", width=400, height=200)
        wc_collision = Wordcloud(
            width=400,
            height=200,
            font_path=font_path,
            place_strategy="random",
            collision_detector=detector,
        )
        wc_collision.generate(sample_text).draw_image(save_file=True, image_name="10_custom_plugins_collision")
    finally:
        unregister_placement_strategy("top_left")
        unregister_collision_detector("simple")


if __name__ == "__main__":
    main()
