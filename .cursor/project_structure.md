## Project structure

### Top level

- **`wordcloud/`**: main library package.
- **`tests/`**: pytest tests for the main package.
- **`examples/`**: small example scripts.
- **`fonts/`**: bundled fonts used by examples and tests.

### `wordcloud/`

- **`wordcloud/__init__.py`**: exports the public API (what users import).
- **`wordcloud/wordcloud.py`**: main `Wordcloud` class (generate, place, draw, export).
- **`wordcloud/cli.py`**: command line interface.
- **`wordcloud/api.py`**: optional Flask API server.

### `wordcloud/utils/`

- **`integral_image.py`**: integral image + placement strategies.
- **`placement.py`**: small placement helper functions.
- **`text_processing.py`**: tokenizing and frequency extraction.
- **`mask.py`**: mask support (allowed and blocked areas).
- **`font_utils.py`**: font loading and caching.
- **`visualization.py`**: color themes and color generation.
- **`export.py`**: PNG, SVG, PDF, GIF, HTML export helpers.
- **`logging_config.py`**: library logging setup.
- **`performance.py`**: performance tracking tools.
- **`trace_utils.py`**: tracing/debug helpers.
- **`collision.py`**: collision detector utilities.
- **`config.py`**: config loading and defaults.


