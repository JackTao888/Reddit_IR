"""Canonical ranker names (BM25 family, TF-IDF family with several similarities,
two-stage hybrid) + factory.

Default matrices omit ``tfidf_overlap`` (poor on long-document Reddit retrieval);
it remains buildable for explicit ``--rankers`` ablations. **LSI** needs SciPy/NumPy
(see ``requirements.txt``).

Legacy aliases ``tfidf`` / ``bm25`` are accepted for scripts and qrels rows
that still use the old names.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

from ..index.store import IndexArtifacts
from .base import BaseRanker
from .bm25_ranker import DEFAULT_B, DEFAULT_FIELD_WEIGHTS, DEFAULT_K1, Bm25Ranker
from .tfidf_ranker import TfidfRanker
from .two_stage_ranker import TwoStageBm25TfidfRanker

_DEFAULT_LSI_K = 100
_DEFAULT_LSI_MAX_VOCAB = 8000
_DEFAULT_PRF_DEPTH = 5
_DEFAULT_PRF_EXPANSION = 15

# Public ordering: TF-IDF family, BM25 family, two-stage hybrid.
# ``tfidf_overlap`` is implemented but omitted here: on long Reddit bodies,
# min(sum_q, sum_d) in the denominator collapses scores (your eval showed
# sharp MAP drop). Pass ``--rankers tfidf_overlap`` explicitly for ablations.
DEFAULT_RANKER_NAMES: Tuple[str, ...] = (
    "tfidf_cosine",
    "tfidf_dice",
    "tfidf_jaccard",
    "bm25_plain",
    "bm25_field",
    "twostage_bm25_tfidf",
    "lsi",
    "bm25_prf",
)

# Names accepted by ``build_ranker`` / ``is_known_ranker`` (defaults + optional).
_SUPPORTED_RANKER_NAMES: frozenset[str] = frozenset((*DEFAULT_RANKER_NAMES, "tfidf_overlap"))

_ALIASES: Dict[str, str] = {
    "tfidf": "tfidf_cosine",
    "bm25": "bm25_plain",
}

# argparse ``choices=`` / help text: supported names plus legacy spellings.
CLI_RANKER_CHOICES: Tuple[str, ...] = tuple(
    sorted(_SUPPORTED_RANKER_NAMES | set(_ALIASES.keys()))
)


def canonical_ranker_name(name: str) -> str:
    """Map legacy CLI names to canonical names."""
    return _ALIASES.get(name, name)


def valid_ranker_names() -> Tuple[str, ...]:
    return DEFAULT_RANKER_NAMES


def is_known_ranker(name: str) -> bool:
    return canonical_ranker_name(name) in _SUPPORTED_RANKER_NAMES


def build_ranker(
    name: str,
    artifacts: IndexArtifacts,
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    field_weights: Optional[Mapping[str, float]] = None,
    stage1_k: int = TwoStageBm25TfidfRanker.DEFAULT_STAGE1_K,
    lsi_k: int = _DEFAULT_LSI_K,
    lsi_max_vocab: int = _DEFAULT_LSI_MAX_VOCAB,
    prf_depth: int = _DEFAULT_PRF_DEPTH,
    prf_expansion_terms: int = _DEFAULT_PRF_EXPANSION,
) -> BaseRanker:
    """Construct a ranker by canonical or legacy name.

    ``field_weights`` is only used when ``name`` canonicalizes to
    ``bm25_field`` (or you pass a custom mapping for extensions).
    """
    n = canonical_ranker_name(name)
    if n == "tfidf_cosine":
        return TfidfRanker(artifacts, similarity="cosine")
    if n == "tfidf_dice":
        return TfidfRanker(artifacts, similarity="dice")
    if n == "tfidf_jaccard":
        return TfidfRanker(artifacts, similarity="jaccard")
    if n == "tfidf_overlap":
        return TfidfRanker(artifacts, similarity="overlap")
    if n == "bm25_plain":
        return Bm25Ranker(artifacts, k1=k1, b=b, field_weights=None)
    if n == "bm25_field":
        fw = field_weights if field_weights is not None else DEFAULT_FIELD_WEIGHTS
        return Bm25Ranker(artifacts, k1=k1, b=b, field_weights=fw)
    if n == "twostage_bm25_tfidf":
        return TwoStageBm25TfidfRanker(artifacts, stage1_k=stage1_k, k1=k1, b=b)
    if n == "lsi":
        from .lsi_ranker import LsiRanker

        return LsiRanker(artifacts, k=lsi_k, max_vocab=lsi_max_vocab)
    if n == "bm25_prf":
        from .prf_ranker import PrfBm25Ranker

        return PrfBm25Ranker(
            artifacts,
            prf_depth=prf_depth,
            expansion_terms=prf_expansion_terms,
            k1=k1,
            b=b,
        )
    raise ValueError(f"Unknown ranker: {name!r} (canonical: {n!r})")
