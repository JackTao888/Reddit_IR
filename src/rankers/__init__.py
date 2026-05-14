"""Ranking models for the JHU 666 IR final project.

Public API:
- RankResult: typed search result row
- BaseRanker: shared query preparation + result formatting
- TfidfRanker: TF-IDF cosine similarity (from scratch on the M3 index)
- Bm25Ranker: BM25 (plain or field-aware)
- prepare_query: query tokenization helper
- main: CLI entry point (``python -m src.rankers.cli``)
"""

from .base import BaseRanker, RankResult
from .bm25_ranker import Bm25Ranker
from .query import prepare_query
from .tfidf_ranker import TfidfRanker

__all__ = [
    "RankResult",
    "BaseRanker",
    "TfidfRanker",
    "Bm25Ranker",
    "prepare_query",
]
