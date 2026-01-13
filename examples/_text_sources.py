from __future__ import annotations

from pathlib import Path

from _bootstrap import ensure_repo_on_path


def load_example_text(name: str = "lorem.txt") -> str:
    """
    Load text from test_sources/ for examples.

    Falls back to a short built-in text if the file is missing.
    """
    repo_root = ensure_repo_on_path()
    path = repo_root / "test_sources" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "wordcloud example text " * 200


