# Project Overview

This project is a flexible Python library for generating word clouds with
customizable placement strategies, styling, and export formats.

## Goals
- Provide a clean Python API for generating word clouds from text.
- Offer multiple placement strategies with consistent interfaces.
- Support common export formats (PNG, SVG, HTML; PDF/GIF optional).
- Make performance profiling and benchmarking repeatable.

## Non-Goals
- Real-time interactive editing in the browser.
- Full NLP pipeline (the text processor is lightweight by design).
- Perfect layout optimality; trade-offs prioritize speed and usability.

## Design Constraints
- Optional dependencies should stay optional.
- The core library must run on macOS, Linux, and Windows.
- Outputs must be reproducible given the same inputs and random seed.

## Primary Use Cases
- Generating static word cloud images for reports or dashboards.
- Comparing layout strategies for research or teaching.
- Prototyping text visualization workflows.

## Audience
- Data analysts and researchers who need quick visual summaries.
- Developers integrating word clouds into scripts or pipelines.
