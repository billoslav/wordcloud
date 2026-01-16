# Testing

## Run the Test Suite
From the repository root:
- `pytest -q`

## Optional Dependencies
Some tests are skipped when optional dependencies are not installed:
- `Pillow` for image export tests.
- `Flask` for API tests.
- `ReportLab` for PDF export tests.

To run all tests, install the optional dependencies listed in `requirements.txt`
or follow the notes in `docs/benchmarks.md`.

## Coverage Focus
Prioritize tests for:
- Placement strategies and collision handling.
- Orientation and rotated text paths.
- Mask handling and dimension alignment.
- Export formats and error handling.

## Debugging Tips
- Use `-k <pattern>` to select a subset of tests.
- Enable logging to see detailed placement behavior.
