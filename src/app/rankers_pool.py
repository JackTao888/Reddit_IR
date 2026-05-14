"""Lazy-init container holding one instance of each available ranker.

Built once at app startup so the TF-IDF doc-norm precomputation isn't
repeated per request.
"""

from __future__ import annotations

from typing import Dict, List

from ..index.store import IndexArtifacts
from ..rankers.base import BaseRanker
from ..rankers.bm25_ranker import DEFAULT_FIELD_WEIGHTS, Bm25Ranker
from ..rankers.tfidf_ranker import TfidfRanker


_AVAILABLE = ("tfidf", "bm25", "bm25_field")


class RankerPool:
    def __init__(self, artifacts: IndexArtifacts):
        self.artifacts = artifacts
        self._cache: Dict[str, BaseRanker] = {}

    def get(self, name: str) -> BaseRanker:
        if name not in _AVAILABLE:
            raise KeyError(name)
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        if name == "tfidf":
            ranker: BaseRanker = TfidfRanker(self.artifacts)
        elif name == "bm25":
            ranker = Bm25Ranker(self.artifacts)
        else:  # bm25_field
            ranker = Bm25Ranker(self.artifacts, field_weights=DEFAULT_FIELD_WEIGHTS)

        self._cache[name] = ranker
        return ranker

    @property
    def available(self) -> List[str]:
        return list(_AVAILABLE)

    def warmup(self) -> None:
        """Eagerly construct every ranker (e.g., before serving traffic)."""
        for name in _AVAILABLE:
            self.get(name)
