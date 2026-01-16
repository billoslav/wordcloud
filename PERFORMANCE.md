# Performance

This document summarizes performance goals and how to measure them.

## Benchmarks
Benchmark scripts live in `benchmarks/` and write logs to `benchmarks/outputs/`.
Run from the repo root:
- `python benchmarks/bench_place_strategies.py`
- `python benchmarks/bench_rotation_overhead.py`
- `python benchmarks/bench_exports.py`

Benchmark summaries are maintained in `docs/benchmarks.md` and sourced from
the logs in `benchmarks/outputs/`.

## Performance Tracking
The `PerformanceTracker` utility in `wordcloud/utils/performance.py` can be
enabled via `Wordcloud(enable_performance_tracking=True)`. Use it to capture
basic runtime information or detailed memory profiling.

## Cache Safety
The cache helpers live in `wordcloud/utils/performance_optimizations.py`.
They support size and age limits, plus disk cleanup.

Default cache settings live under `performance.cache` in the config.
Key fields include `lru_max_size`, `lru_max_age_seconds`, `disk_max_entries`,
`disk_max_size_bytes`, `disk_max_age_seconds`, and `disk_cleanup_interval_seconds`.

## Streaming Inputs
For large corpora, use `Wordcloud.generate_from_stream()` with text chunks.
This avoids loading all text into memory at once.

## Profiling Tips
- Use smaller input texts for quick iteration.
- Pin random seeds when comparing runs.
- Keep the same font and mask assets for stable comparisons.
