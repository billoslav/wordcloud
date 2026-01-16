Examples for using the Wordcloud library.

Run them from the project root. Outputs are written to `Results/`.
Pre-generated outputs are committed under `Results/` so you can browse without running the scripts.

All examples load their input text from `test_sources/` (for consistent results).

### Pre-generated outputs

- `00_quickstart.py` -> `Results/00_quickstart.png`
- `01_stopwords_and_min_word_length.py` -> `Results/01_stopwords_and_min_word_length.png`
- `02_color_themes.py` -> `Results/02_color_theme_default.png`, `Results/02_color_theme_plasma.png`, `Results/02_color_theme_viridis.png`
- `03_rotation.py` -> `Results/03_rotation_mostly_horizontal.png`, `Results/03_rotation_more_vertical.png`
- `04_masks.py` -> `Results/04_mask_circle.png`
- `05_export_svg_and_html.py` -> `Results/05_wordcloud.svg`, `Results/05_wordcloud.html`
- `08_performance_tracking.py` -> `Results/08_performance_tracking.png`
- `09_placement_strategies_gallery.py` -> `Results/09_strategy_*.png`
- `basic_wordcloud.py` -> `Results/example_basic.png`
- `mask_wordcloud.py` -> `Results/example_masked.png`

Additional benchmark and gallery outputs are available in `Results/` (e.g., `09_strategy_*.png` and `bench.*`).

### Start here

```bash
python examples/00_quickstart.py
```

### Core features

- **Text filtering**

```bash
python examples/01_stopwords_and_min_word_length.py
```

- **Color themes**

```bash
python examples/02_color_themes.py
```

- **Rotation / orientation**

```bash
python examples/03_rotation.py
```

- **Masks**

```bash
python examples/04_masks.py
```

### Exports

- **SVG + HTML**

```bash
python examples/05_export_svg_and_html.py
```

- **PDF + GIF (optional dependencies)**

```bash
python examples/06_export_pdf_and_gif_optional.py
```

### API (optional dependency)

```bash
python examples/07_api_optional.py
```

### Performance tracking

```bash
python examples/08_performance_tracking.py
```

### Strategy gallery

```bash
python examples/09_placement_strategies_gallery.py
```

### Older examples

```bash
python examples/basic_wordcloud.py
python examples/mask_wordcloud.py
```

