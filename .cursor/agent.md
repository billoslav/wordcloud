## Agent notes (Wordcloud)

### What this project is

This repo is a Python library that generates wordcloud images from text.

Key pieces:
- `wordcloud/wordcloud.py`: main `Wordcloud` class and public behavior.
- `wordcloud/utils/`: placement, masks, exports, logging, performance tools.
- `tests/`: test suite (pytest).
- `Wordcloud` supports deterministic output via `random_state`.

### How to run it

Run tests:
"""
bash
pytest
"""

Optional perf checks:
"""
bash
WORDCLOUD_PERF_BUDGETS=1 pytest -m performance
"""

Run the CLI:
"""
bash
python -m wordcloud.cli --text "hello world" --output Results/hello.png
"""

Run the API (requires Flask installed):
"""
bash
python -c "from wordcloud.api import run_api; run_api(port=5001)"
"""

### Dev rules we follow here

- Keep code stable first, then optimize.
- Prefer clear errors over silent failures.
- Add or update tests with every behavior change.
- Avoid duplicated implementations (goal: single `wordcloud/` implementation).


