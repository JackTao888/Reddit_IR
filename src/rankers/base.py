"""Shared base class and result types for rankers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from ..index.store import IndexArtifacts
from .query import prepare_query


@dataclass
class RankResult:
    rank: int
    doc_id: str
    score: float


class BaseRanker:
    name: str = "base"

    def __init__(self, artifacts: IndexArtifacts):
        self.artifacts = artifacts
        self._preprocessing = artifacts.preprocessing or {}

    def _prepare_query(self, query: str) -> List[str]:
        return prepare_query(query, self._preprocessing)

    @staticmethod
    def _format_results(
        scored: Iterable[Tuple[str, float]],
        top_k: int,
    ) -> List[RankResult]:
        ranked = sorted(scored, key=lambda x: -x[1])[:top_k]
        return [
            RankResult(rank=i + 1, doc_id=doc_id, score=score)
            for i, (doc_id, score) in enumerate(ranked)
        ]

    def search(self, query: str, *, top_k: int = 10) -> List[RankResult]:
        raise NotImplementedError
