## Benchmarks

This folder contains small scripts that measure runtime of different parts of Wordcloud.

### Run benchmarks
From the repo root:

"""
python benchmarks/bench_place_strategies.py
python benchmarks/bench_rotation_overhead.py
python benchmarks/bench_exports.py
"""

Note: benchmark scripts can also be run from any working directory. They add the repo root to `sys.path` automatically.

### Latest Results (2026-01-16)
Source logs: `benchmarks/outputs/`.

#### Placement strategies
"""
random               10.1272s  placed_words=63
brute                 1.8489s  placed_words=63
archimedian          10.4498s  placed_words=63
rectangular          10.6921s  placed_words=63
archimedian_reverse  10.9480s  placed_words=63
rectangular_reverse  11.0792s  placed_words=63
KDTree               16.5685s  placed_words=63
quad                133.6024s  placed_words=63
pytag                11.2977s  placed_words=63
pytag_reverse        11.2614s  placed_words=63
circular              6.8190s  placed_words=0
hierarchical          6.9130s  placed_words=0
grid                 14.6552s  placed_words=63
force_directed       15.1069s  placed_words=63
"""

#### Rotation overhead (single run each)
"""
prefer_horizontal=1.0   10.8237s
prefer_horizontal=0.9   10.6218s
prefer_horizontal=0.7   10.3056s
prefer_horizontal=0.5   10.3361s
"""

#### Exports
"""
generate(): 10.1711s
PNG save:   0.0533s
SVG gen:    0.0664s
HTML gen:   0.1299s
PDF save:   skipped (install reportlab)
GIF save:   0.0482s
"""

### Notes
- Benchmarks use a fixed seed so runs are repeatable.
- Benchmarks write output files into `Results/` when needed.
- Some exports need extra dependencies:
  - PDF: install `reportlab`
  - API: install `flask`

### Performance regression checks
Record a baseline:

"""
python benchmarks/perf_regression.py --record
"""

Check against the baseline:

"""
python benchmarks/perf_regression.py --check
"""

Run the opt-in pytest check:

"""
WORDCLOUD_PERF_BUDGETS=1 pytest -q -m performance
"""

### CI-friendly test commands
Fast tests:

"""
pytest -q
"""

If you later add slow benchmarks as tests, mark them and run them separately:

"""
pytest -q -m "not slow"
pytest -q -m slow
"""


