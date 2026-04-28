"""Inverted-index package for the JHU 666 IR final project.

Public entry points:
- IndexConfig: typed build configuration
- InvertedIndex: per-field term-doc-frequency structure
- DocStore / DocMetadata: per-document metadata for ranking + display
- IndexArtifacts: the persistable bundle (indexes + docs + manifest)
- build_artifacts: build the bundle from preprocessed JSONL
- save_artifacts / load_artifacts: persistence helpers
- main: CLI entry point (``python -m src.index.cli``)
"""

from .builder import build_artifacts
from .config import IndexConfig
from .inverted import InvertedIndex
from .store import DocMetadata, DocStore, IndexArtifacts, load_artifacts, save_artifacts

__all__ = [
    "IndexConfig",
    "InvertedIndex",
    "DocMetadata",
    "DocStore",
    "IndexArtifacts",
    "build_artifacts",
    "save_artifacts",
    "load_artifacts",
]
