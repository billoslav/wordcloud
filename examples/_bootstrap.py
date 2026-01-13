from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_on_path() -> Path:
    """
    Ensure the repo root is on sys.path so `import wordcloud` works when running
    examples directly (python examples/xx.py).

    Returns the repo root path.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


