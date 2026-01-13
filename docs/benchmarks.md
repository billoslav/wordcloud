## Benchmarks

This folder contains small scripts that measure runtime of different parts of Wordcloud.

### Run benchmarks

From the repo root:

"""
bash
python benchmarks/bench_place_strategies.py
python benchmarks/bench_rotation_overhead.py
python benchmarks/bench_exports.py
"""

Note: benchmark scripts can also be run from any working directory. They add the repo root to `sys.path` automatically.

### Notes

- Benchmarks write output files into `Results/` when needed.
- Some exports need extra dependencies:
  - PDF: install `reportlab`
  - API: install `flask`

### CI-friendly test commands

Fast tests:

"""
bash
pytest -q
"""

If you later add slow benchmarks as tests, mark them and run them separately:

"""
bash
pytest -q -m "not slow"
pytest -q -m slow
"""


