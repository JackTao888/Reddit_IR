"""Lazy-init container holding one instance of each available ranker.

Built once at app startup so the TF-IDF doc-norm precomputation isn't
repeated per request.
"""

from __future__ import annotations

from typing import Dict, List

from ..index.store import IndexArtifacts
from ..rankers.base import BaseRanker
from ..rankers.registry import DEFAULT_RANKER_NAMES, build_ranker


class RankerPool:
    def __init__(self, artifacts: IndexArtifacts):
        self.artifacts = artifacts
        self._cache: Dict[str, BaseRanker] = {}

    def get(self, name: str) -> BaseRanker:
        if name not in DEFAULT_RANKER_NAMES:
            raise KeyError(name)
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        ranker = build_ranker(name, self.artifacts)
        self._cache[name] = ranker
        return ranker

    @property
    def available(self) -> List[str]:
        return list(DEFAULT_RANKER_NAMES)

    def warmup(self) -> None:
        """Eagerly construct every ranker (e.g., before serving traffic)."""
        for name in DEFAULT_RANKER_NAMES:
            self.get(name)
