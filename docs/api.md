# API Overview

This document summarizes the public API surface of the project.

## Python API
### `Wordcloud`
Primary entry point for generation.

Common usage:
- `Wordcloud(...).generate(text).draw_image()`
- `Wordcloud(...).generate(text).generate_svg()`
- `Wordcloud(...).generate(text).create_html(svg_content)`
- `Wordcloud(...).generate(text).create_html(interactive=True)` for zoom, pan, and search
- `Wordcloud(...).generate_from_stream(chunks)` for large inputs

Notable configuration:
- Placement: `place_strategy`, `collision_detector`
- Orientation: `prefer_horizontal`, `rotation_angles`
- Masking: `mask_image`, `mask_threshold`
- Performance: `enable_performance_tracking`, `performance_tracking_detail`
- Determinism: `random_state`

## CLI
Entry point: `wordcloud/cli.py`
- `wordcloud --text "..." --output Results/example.png --width 600 --height 400`
- `wordcloud --file input.txt --formats png,svg,html --output Results/example`

## Optional API Server
Module: `wordcloud/api.py`
- `run_api(port=5001)` to start the Flask server.
- `POST /api/generate` accepts `formats` (list or comma string).
- `GET /api/download/<job_id>?format=svg` downloads a chosen format.

## SVG Export
- SVG text uses kerning-aware metrics and baseline alignment.
- `textLength` is set to match PIL font measurements.

## Extensions
- `wordcloud.utils.placement.register_placement_strategy()` registers custom placement.
- `wordcloud.utils.collision.register_collision_detector()` registers custom collision detectors.

## Stability Notes
- The `Wordcloud` class is considered stable.
- Utility modules under `wordcloud/utils/` are internal and may change.

## GPU Acceleration
- GPU support is opt-in and off by default.
- Use `create_gpu_accelerator(enabled=True)` to try GPU paths.
- Requires `cupy` (CUDA) or `pyopencl` (OpenCL).
- GPU results are validated and fall back to CPU on mismatch.
