# Documentation Overview

This folder contains the canonical documentation set for the project.
Start with `DOCS_INDEX.md` at the repository root for the complete map.

## Contents
- `benchmarks.md` — benchmark methodology and summary outputs.
- `examples.md` — curated example list and expected outputs.
- `api.md` — public API overview and stability notes.

## Repository Structure (High Level)
```
wordcloud/
├── wordcloud/            # Core package
├── examples/             # Example scripts and README
├── benchmarks/           # Benchmark scripts and logs
├── docs/                 # Documentation (this folder)
├── tests/                # Test suite
├── fonts/                # Fonts used by examples
├── Results/              # Generated example outputs
├── Tracking/             # Tracing outputs
├── test_sources/         # Input text fixtures
├── README.md             # Project overview
└── ARCHITECTURE.md       # Architecture summary
```

## Documentation Guidelines
- Keep docs concise and action-oriented.
- Use consistent terminology with code (module/class names).
- Prefer links to source files over copying code.
- Update `DOCS_INDEX.md` when adding new docs.
