"""Ranking models for the JHU 666 IR final project.

Public API:
- RankResult: typed search result row
- BaseRanker: shared query preparation + result formatting
- TfidfRanker: TF-IDF cosine similarity (from scratch on the M3 index)
- Bm25Ranker: BM25 (plain or field-aware)
- TwoStageBm25TfidfRanker: BM25-plain recall then TF-IDF cosine rerank
- LsiRanker / PrfBm25Ranker: see ``lsi_ranker.py`` and ``prf_ranker.py``; construct via ``build_ranker("lsi")`` / ``build_ranker("bm25_prf")`` (LSI needs SciPy).
- build_ranker / DEFAULT_RANKER_NAMES: registry for CLI, UI, and eval
- prepare_query: query tokenization helper
- main: CLI entry point (``python -m src.rankers.cli``)
"""

from .base import BaseRanker, RankResult
from .bm25_ranker import Bm25Ranker
from .query import prepare_query
from .registry import (
    CLI_RANKER_CHOICES,
    DEFAULT_RANKER_NAMES,
    build_ranker,
    canonical_ranker_name,
    is_known_ranker,
    valid_ranker_names,
)
from .tfidf_ranker import TfidfRanker
from .two_stage_ranker import TwoStageBm25TfidfRanker

__all__ = [
    "RankResult",
    "BaseRanker",
    "TfidfRanker",
    "Bm25Ranker",
    "TwoStageBm25TfidfRanker",
    "prepare_query",
    "DEFAULT_RANKER_NAMES",
    "CLI_RANKER_CHOICES",
    "build_ranker",
    "canonical_ranker_name",
    "is_known_ranker",
    "valid_ranker_names",
]
