"""Preprocessing package for the JHU 666 IR final project.

Public entry points:
- PreprocessConfig: typed run configuration
- preprocess_doc: transform a single crawler record
- run_preprocess: process all subreddit JSONL files end-to-end
- main: CLI entry point (``python -m src.preprocess.cli``)
"""

from .config import PreprocessConfig
from .pipeline import preprocess_doc, run_preprocess

__all__ = ["PreprocessConfig", "preprocess_doc", "run_preprocess"]
