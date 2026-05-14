"""Pseudo-relevance feedback (PRF) on top of BM25.

Classic **Rocchio-style** idea for ad-hoc IR (still lexical, but uses *implicit*
relevance from a first-pass retrieval): run BM25, pool text from the top-R
hits, extract high-frequency terms from that pool with the **same**
preprocessing as the index, append them to the query string, and run BM25
again. Surfaces vocabulary that co-occurs with query terms in likely-relevant
snippets without any embedding model.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from ..index.store import IndexArtifacts
from .base import BaseRanker, RankResult
from .bm25_ranker import DEFAULT_B, DEFAULT_K1, Bm25Ranker


class PrfBm25Ranker(BaseRanker):
    """Two-pass BM25: first retrieval feeds term expansion for the second pass."""

    name = "bm25_prf"

    def __init__(
        self,
        artifacts: IndexArtifacts,
        *,
        prf_depth: int = 5,
        expansion_terms: int = 15,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        super().__init__(artifacts)
        self.prf_depth = max(1, int(prf_depth))
        self.expansion_terms = max(1, int(expansion_terms))
        self._inner = Bm25Ranker(artifacts, k1=k1, b=b, field_weights=None)

    def search(self, query: str, *, top_k: int = 10) -> List[RankResult]:
        tokens = self._prepare_query(query)
        if not tokens:
            return []

        first = self._inner.search(query, top_k=self.prf_depth)
        if not first:
            return []

        chunks: List[str] = []
        for r in first:
            meta = self.artifacts.doc_store.get(r.doc_id)
            if meta is None:
                continue
            chunks.append(f"{meta.title} {meta.selftext_excerpt}")

        blob = " ".join(chunks)
        extra_tokens = self._prepare_query(blob)
        qset = set(tokens)
        counts = Counter(t for t in extra_tokens if t not in qset)
        additions = [t for t, _ in counts.most_common(self.expansion_terms)]
        if not additions:
            return self._inner.search(query, top_k=top_k)

        expanded = " ".join(tokens + additions)
        return self._inner.search(expanded, top_k=top_k)
