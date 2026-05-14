"""TF-IDF vector ranker with multiple similarity functions (HW2-style).

Built on the M3 inverted index. Default is **cosine** on L2-normalized TF-IDF
weights; alternatives use the same term weights with **Dice**, **Jaccard**, or
**Overlap** on the non-negative weighted vectors (sums use full document mass,
not only query-matched terms — precomputed per doc).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Literal, Sequence

from ..index.store import IndexArtifacts
from .base import BaseRanker, RankResult

TfidfSimilarity = Literal["cosine", "dice", "jaccard", "overlap"]


class TfidfRanker(BaseRanker):
    """TF-IDF term weights with configurable vector similarity."""

    def __init__(
        self,
        artifacts: IndexArtifacts,
        *,
        similarity: TfidfSimilarity = "cosine",
    ):
        super().__init__(artifacts)
        if similarity not in ("cosine", "dice", "jaccard", "overlap"):
            raise ValueError(f"similarity must be cosine|dice|jaccard|overlap, got {similarity!r}")
        self._similarity: TfidfSimilarity = similarity
        self.name = "tfidf_cosine" if similarity == "cosine" else f"tfidf_{similarity}"

        self._idx = artifacts.all_index

        n = self._idx.n_docs
        self._idf: Dict[str, float] = {}
        for term, df in self._idx.df.items():
            if df > 0 and n > 0:
                self._idf[term] = math.log(n / df) if n > df else 0.0

        self._doc_norms = self._compute_doc_norms()
        self._doc_l1: Dict[str, float] = (
            {} if similarity == "cosine" else self._compute_doc_l1()
        )

    def _doc_term_weight(self, tf: int, idf: float) -> float:
        return (1.0 + math.log(tf)) * idf if tf > 0 else 0.0

    def _compute_doc_norms(self) -> Dict[str, float]:
        sums: Dict[str, float] = defaultdict(float)
        for term, postings in self._idx.term_to_postings.items():
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for doc_id, tf in postings.items():
                w = self._doc_term_weight(tf, idf)
                sums[doc_id] += w * w
        return {d: math.sqrt(s) for d, s in sums.items()}

    def _compute_doc_l1(self) -> Dict[str, float]:
        """Sum of TF-IDF weights per document (for Dice / Jaccard / Overlap)."""
        sums: Dict[str, float] = defaultdict(float)
        for term, postings in self._idx.term_to_postings.items():
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for doc_id, tf in postings.items():
                sums[doc_id] += self._doc_term_weight(tf, idf)
        return dict(sums)

    def _similarity_score(
        self,
        *,
        doc_id: str,
        dot: float,
        q_norm_l2: float,
        sum_q_l1: float,
    ) -> float:
        if self._similarity == "cosine":
            d_norm = self._doc_norms.get(doc_id, 0.0)
            denom = q_norm_l2 * d_norm
            return dot / denom if denom > 0.0 else 0.0

        sum_d = self._doc_l1.get(doc_id, 0.0)
        if self._similarity == "dice":
            denom = sum_q_l1 + sum_d
            return (2.0 * dot / denom) if denom > 0.0 else 0.0
        if self._similarity == "jaccard":
            denom = sum_q_l1 + sum_d - dot
            return (dot / denom) if denom > 0.0 else 0.0
        if self._similarity == "overlap":
            denom = min(sum_q_l1, sum_d)
            return (dot / denom) if denom > 0.0 else 0.0
        return 0.0

    def search(self, query: str, *, top_k: int = 10) -> List[RankResult]:
        tokens = self._prepare_query(query)
        if not tokens:
            return []

        q_tf = Counter(tokens)
        q_weights: Dict[str, float] = {}
        for term, count in q_tf.items():
            idf = self._idf.get(term)
            if idf is None or idf == 0.0:
                continue
            q_weights[term] = self._doc_term_weight(count, idf)

        if not q_weights:
            return []

        q_norm_l2 = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm_l2 == 0.0:
            return []

        sum_q_l1 = sum(q_weights.values())

        dot_products: Dict[str, float] = defaultdict(float)
        for term, q_w in q_weights.items():
            idf = self._idf[term]
            postings = self._idx.term_to_postings.get(term, {})
            for doc_id, tf in postings.items():
                d_w = self._doc_term_weight(tf, idf)
                dot_products[doc_id] += q_w * d_w

        scored = []
        for doc_id, dot in dot_products.items():
            s = self._similarity_score(
                doc_id=doc_id,
                dot=dot,
                q_norm_l2=q_norm_l2,
                sum_q_l1=sum_q_l1,
            )
            scored.append((doc_id, s))

        return self._format_results(scored, top_k)

    def rerank_candidates(
        self,
        query: str,
        doc_ids: Sequence[str],
        *,
        top_k: int,
    ) -> List[RankResult]:
        """Rerank candidates using this ranker's configured similarity (two-stage uses cosine)."""
        tokens = self._prepare_query(query)
        if not tokens or not doc_ids:
            return []

        q_tf = Counter(tokens)
        q_weights: Dict[str, float] = {}
        for term, count in q_tf.items():
            idf = self._idf.get(term)
            if idf is None or idf == 0.0:
                continue
            q_weights[term] = self._doc_term_weight(count, idf)

        if not q_weights:
            return []

        q_norm_l2 = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm_l2 == 0.0:
            return []

        sum_q_l1 = sum(q_weights.values())

        seen: Dict[str, int] = {}
        unique: List[str] = []
        for i, d in enumerate(doc_ids):
            if d not in seen:
                seen[d] = i
                unique.append(d)

        dots: Dict[str, float] = defaultdict(float)
        for term, q_w in q_weights.items():
            idf = self._idf[term]
            postings = self._idx.term_to_postings.get(term, {})
            for doc_id in unique:
                tf = postings.get(doc_id)
                if tf:
                    d_w = self._doc_term_weight(tf, idf)
                    dots[doc_id] += q_w * d_w

        scored: List[tuple] = []
        for doc_id in unique:
            dot = dots[doc_id]
            sim = self._similarity_score(
                doc_id=doc_id,
                dot=dot,
                q_norm_l2=q_norm_l2,
                sum_q_l1=sum_q_l1,
            )
            scored.append((doc_id, sim, seen[doc_id]))

        scored.sort(key=lambda x: (-x[1], x[2]))
        trimmed = [(doc_id, s) for doc_id, s, _ in scored[:top_k]]
        return self._format_results(trimmed, top_k)
