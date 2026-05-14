"""BM25 ranker (plain and field-aware).

Plain mode scores against the ``all`` index. Field-aware mode runs the same
formula on per-field indexes (``title``, ``body``, ``comments``) and combines
them with caller-provided weights.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional

from ..index.inverted import InvertedIndex
from ..index.store import IndexArtifacts
from .base import BaseRanker, RankResult


DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_FIELD_WEIGHTS: Dict[str, float] = {"title": 2.0, "body": 1.0, "comments": 1.2}


class Bm25Ranker(BaseRanker):
    def __init__(
        self,
        artifacts: IndexArtifacts,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        field_weights: Optional[Mapping[str, float]] = None,
    ):
        super().__init__(artifacts)
        self.k1 = float(k1)
        self.b = float(b)
        # Defensive copy + drop zero/negative weights so they're truly disabled.
        self.field_weights: Optional[Dict[str, float]] = (
            {k: float(v) for k, v in field_weights.items() if float(v) > 0.0}
            if field_weights is not None
            else None
        )
        self.name = (
            f"bm25_field(k1={self.k1},b={self.b})"
            if self.field_weights is not None
            else f"bm25(k1={self.k1},b={self.b})"
        )

    @staticmethod
    def _idf(n_docs: int, df: int) -> float:
        # Lucene-style BM25 IDF: always >= 0, monotone in df.
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _score_field(
        self,
        idx: InvertedIndex,
        query_tf: Mapping[str, int],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        n_docs = idx.n_docs
        avgdl = idx.avgdl or 1.0

        for term in query_tf:
            df = idx.df.get(term, 0)
            if df == 0 or n_docs == 0:
                continue
            idf = self._idf(n_docs, df)
            postings = idx.term_to_postings.get(term, {})
            for doc_id, tf in postings.items():
                doc_len = idx.doc_lens.get(doc_id, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
                if denom > 0.0:
                    scores[doc_id] += idf * tf * (self.k1 + 1.0) / denom
        return scores

    def search(self, query: str, *, top_k: int = 10) -> List[RankResult]:
        tokens = self._prepare_query(query)
        if not tokens:
            return []

        query_tf = Counter(tokens)

        if self.field_weights is None:
            scored = self._score_field(self.artifacts.all_index, query_tf)
            return self._format_results(scored.items(), top_k)

        combined: Dict[str, float] = defaultdict(float)
        for field_name, weight in self.field_weights.items():
            field_idx = self.artifacts.indexes.get(field_name)
            if field_idx is None:
                continue
            field_scores = self._score_field(field_idx, query_tf)
            for doc_id, s in field_scores.items():
                combined[doc_id] += weight * s

        return self._format_results(combined.items(), top_k)
