"""TF-IDF cosine similarity ranker.

Built directly on top of the M3 inverted index so we don't need sklearn.
Document norms are precomputed once at construction; per-query work scales
with the number of unique terms in the query.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List

from ..index.store import IndexArtifacts
from .base import BaseRanker, RankResult


class TfidfRanker(BaseRanker):
    name = "tfidf"

    def __init__(self, artifacts: IndexArtifacts):
        super().__init__(artifacts)
        self._idx = artifacts.all_index

        n = self._idx.n_docs
        self._idf: Dict[str, float] = {}
        for term, df in self._idx.df.items():
            if df > 0 and n > 0:
                # Standard TF-IDF IDF, no smoothing — fine for our 10k corpus.
                self._idf[term] = math.log(n / df) if n > df else 0.0

        self._doc_norms = self._compute_doc_norms()

    def _doc_term_weight(self, tf: int, idf: float) -> float:
        # Sublinear TF: dampens repeated terms while preserving signal at tf=1,2,3.
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

        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0.0:
            return []

        dot_products: Dict[str, float] = defaultdict(float)
        for term, q_w in q_weights.items():
            idf = self._idf[term]
            postings = self._idx.term_to_postings.get(term, {})
            for doc_id, tf in postings.items():
                d_w = self._doc_term_weight(tf, idf)
                dot_products[doc_id] += q_w * d_w

        scored = []
        for doc_id, dot in dot_products.items():
            d_norm = self._doc_norms.get(doc_id, 0.0)
            denom = q_norm * d_norm
            if denom > 0.0:
                scored.append((doc_id, dot / denom))

        return self._format_results(scored, top_k)
