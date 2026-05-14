"""Two-stage retrieval: BM25 (plain) for recall, TF-IDF cosine for reranking."""

from __future__ import annotations

from typing import List

from ..index.store import IndexArtifacts
from .base import BaseRanker, RankResult
from .bm25_ranker import DEFAULT_B, DEFAULT_K1, Bm25Ranker
from .tfidf_ranker import TfidfRanker


class TwoStageBm25TfidfRanker(BaseRanker):
    """Stage 1: ``bm25_plain`` over the full index. Stage 2: TF-IDF cosine on candidates only."""

    DEFAULT_STAGE1_K = 200
    name = "twostage_bm25_tfidf"

    def __init__(
        self,
        artifacts: IndexArtifacts,
        *,
        stage1_k: int = DEFAULT_STAGE1_K,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        super().__init__(artifacts)
        self.stage1_k = max(1, int(stage1_k))
        self._bm25 = Bm25Ranker(artifacts, k1=k1, b=b, field_weights=None)
        self._tfidf = TfidfRanker(artifacts)

    def search(self, query: str, *, top_k: int = 10) -> List[RankResult]:
        pool = max(self.stage1_k, top_k)
        candidates = self._bm25.search(query, top_k=pool)
        if not candidates:
            return []
        doc_ids = [r.doc_id for r in candidates]
        return self._tfidf.rerank_candidates(query, doc_ids, top_k=top_k)
