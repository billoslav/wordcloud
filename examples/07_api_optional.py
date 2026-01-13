from __future__ import annotations

"""
This example starts the Flask API if Flask is installed.

Usage:
  python examples/07_api_optional.py

Then try:
  curl -X POST http://localhost:5001/api/generate -H "Content-Type: application/json" -d '{"text":"hello api"}'
"""

import sys

from _bootstrap import ensure_repo_on_path


def main() -> None:
    ensure_repo_on_path()
    try:
        import flask  # noqa: F401
    except Exception:
        print("API example skipped. Install extra: flask")
        print("Try: pip install flask")
        return

    from wordcloud.api import run_api

    print("Starting API on port 5001...")
    print("Press Ctrl+C to stop.")
    run_api(host="127.0.0.1", port=5001, debug=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)


