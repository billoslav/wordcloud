# Contributing

Thanks for contributing. This project values small, focused changes with clear
tests and documentation updates.

## Development Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run tests:
   - `pytest -q`
4. Run linter:
   - `ruff check .`

## Workflow
- Keep changes scoped and easy to review.
- Update or add tests for behavior changes.
- Update docs in `README.md` or `docs/` if user-facing behavior changes.

## Code Style
- Follow `STYLE_GUIDE.md`.
- Prefer readable, explicit code over clever shortcuts.
- Use type hints where they improve clarity.

## Pull Request Checklist
- Tests pass locally.
- New or changed behavior has tests.
- Docs updated (if applicable).
- No unrelated formatting churn.
