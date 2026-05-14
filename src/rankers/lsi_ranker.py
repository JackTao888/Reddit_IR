"""LSI / truncated-SVD retrieval (HW2 Extension 9 style).

Uses the same sublinear TF–IDF weights as ``TfidfRanker`` on the ``all`` index,
builds a sparse term–document matrix, runs ``scipy.sparse.linalg.svds``, then
scores by **cosine in the k-dimensional latent space** (query projected with
``Vt @ q``). This captures co-occurrence structure (classical “semantic-ish”
smoothing), not neural embeddings.

Requires **SciPy** (and NumPy). Install: ``pip install scipy``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..index.store import IndexArtifacts
from .base import BaseRanker, RankResult
from .bm25_ranker import Bm25Ranker

try:
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised when scipy missing
    np = None  # type: ignore[assignment]
    csr_matrix = None  # type: ignore[assignment]
    svds = None  # type: ignore[assignment]
    _HAS_SCIPY = False


def _require_scipy() -> None:
    if not _HAS_SCIPY:
        raise ImportError(
            "LsiRanker requires scipy and numpy. Install with: pip install scipy"
        )


def _select_vocab(df: Dict[str, int], max_vocab: int) -> List[str]:
    pairs = sorted(df.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in pairs[:max_vocab]]


class LsiRanker(BaseRanker):
    """Truncated SVD on a TF–IDF-weighted term–document matrix (LSI retrieval)."""

    def __init__(
        self,
        artifacts: IndexArtifacts,
        *,
        k: int = 100,
        max_vocab: int = 8000,
    ):
        _require_scipy()
        super().__init__(artifacts)
        self.k_requested = max(1, int(k))
        self.max_vocab = max(100, int(max_vocab))
        self._fallback: Optional[Bm25Ranker] = None

        self._idx = artifacts.all_index
        n = self._idx.n_docs
        self._idf: Dict[str, float] = {}
        for term, df in self._idx.df.items():
            if df > 0 and n > 0:
                self._idf[term] = math.log(n / df) if n > df else 0.0

        self._vocab = _select_vocab(self._idx.df, self.max_vocab)
        self._term_to_col = {t: j for j, t in enumerate(self._vocab)}
        n_terms = len(self._vocab)

        self._doc_ids: List[str] = sorted(artifacts.doc_store.all_doc_ids())
        self._doc_row = {d: i for i, d in enumerate(self._doc_ids)}
        n_rows = len(self._doc_ids)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        def doc_weight(tf: int, idf: float) -> float:
            return (1.0 + math.log(tf)) * idf if tf > 0 else 0.0

        for term in self._vocab:
            j = self._term_to_col[term]
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            postings = self._idx.term_to_postings.get(term, {})
            for doc_id, tf in postings.items():
                ri = self._doc_row.get(doc_id)
                if ri is None:
                    continue
                w = doc_weight(tf, idf)
                if w != 0.0:
                    rows.append(ri)
                    cols.append(j)
                    data.append(w)

        if n_rows < 2 or n_terms < 2 or not data:
            self._fallback = Bm25Ranker(artifacts)
            self.name = f"lsi(fallback→bm25_plain,max_vocab={self.max_vocab})"
            return

        assert np is not None and csr_matrix is not None and svds is not None
        a = csr_matrix((data, (rows, cols)), shape=(n_rows, n_terms), dtype=np.float64)
        k_use = min(self.k_requested, n_rows - 1, n_terms - 1)
        if k_use < 1:
            self._fallback = Bm25Ranker(artifacts)
            self.name = f"lsi(fallback→bm25_plain,max_vocab={self.max_vocab})"
            return

        try:
            u, s, vt = svds(a, k=k_use, which="LM")
        except Exception:
            self._fallback = Bm25Ranker(artifacts)
            self.name = f"lsi(fallback→bm25_plain,max_vocab={self.max_vocab})"
            return

        u = np.flip(u, axis=1)
        s = np.flip(s)
        vt = np.flip(vt, axis=0)
        doc_latent = u * s
        norms = np.linalg.norm(doc_latent, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        self._doc_unit = (doc_latent / norms).astype(np.float64)
        self._Vt = vt.astype(np.float64)
        self.k_dim = int(vt.shape[0])
        self.name = f"lsi(k={self.k_dim},max_vocab={self.max_vocab})"

    def _query_term_weight(self, tf: int, idf: float) -> float:
        return (1.0 + math.log(tf)) * idf if tf > 0 else 0.0

    def search(self, query: str, *, top_k: int = 10) -> List[RankResult]:
        from collections import Counter

        if self._fallback is not None:
            return self._fallback.search(query, top_k=top_k)

        assert np is not None
        tokens = self._prepare_query(query)
        if not tokens:
            return []

        q_tf = Counter(tokens)
        n_terms = len(self._vocab)
        q_vec = np.zeros(n_terms, dtype=np.float64)
        for term, cnt in q_tf.items():
            j = self._term_to_col.get(term)
            if j is None:
                continue
            idf = self._idf.get(term)
            if idf is None or idf == 0.0:
                continue
            q_vec[j] = self._query_term_weight(cnt, idf)

        q_latent = self._Vt @ q_vec
        qn = float(np.linalg.norm(q_latent))
        if qn < 1e-12:
            return []
        q_unit = (q_latent / qn).astype(np.float64)

        sims = self._doc_unit @ q_unit
        scored: List[Tuple[str, float]] = [
            (self._doc_ids[i], float(sims[i])) for i in range(len(self._doc_ids))
        ]
        return self._format_results(scored, top_k)
