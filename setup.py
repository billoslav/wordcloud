#!/usr/bin/env python3
from setuptools import setup, find_packages # type: ignore
from pathlib import Path

here = Path(__file__).parent
readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

# Minimal setup for local CLI usage; not publishing
setup(
    name="wordcloud-local",
    version="0.0.0",
    description="Local wordcloud package",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["wordcloud", "wordcloud.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pillow>=8.0.0",
        "matplotlib>=3.4.0",
    ],
    entry_points={
        "console_scripts": [
            "wordcloud=wordcloud.cli:main",
        ],
    },
)

