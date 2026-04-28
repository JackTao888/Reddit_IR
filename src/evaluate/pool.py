"""Pool generation: union top-N across rankers, shuffled per query.

Used to produce a labeling template (``pool.csv``) that the user fills in
manually. The labeler is shown title + excerpt only — no scores, no rank,
no ranker identity — so judgments aren't anchored to any specific model.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

from ..index.store import IndexArtifacts
from ..rankers.base import BaseRanker
from .config import DEFAULT_POOL_DEPTH, DEFAULT_SHUFFLE_SEED
from .queries import Query


_EXCERPT_CHARS = 220


def _excerpt(meta) -> str:
    if meta is None:
        return ""
    text = meta.selftext_excerpt or meta.title or ""
    return (text or "")[:_EXCERPT_CHARS].replace("\n", " ").strip()


def generate_pool(
    artifacts: IndexArtifacts,
    rankers: Sequence[Tuple[str, BaseRanker]],
    queries: Iterable[Query],
    *,
    pool_depth: int = DEFAULT_POOL_DEPTH,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> List[dict]:
    rng = random.Random(shuffle_seed)
    rows: List[dict] = []

    for q in queries:
        seen: set = set()
        ordered_doc_ids: List[str] = []
        for _name, ranker in rankers:
            for r in ranker.search(q.text, top_k=pool_depth):
                if r.doc_id not in seen:
                    seen.add(r.doc_id)
                    ordered_doc_ids.append(r.doc_id)

        rng.shuffle(ordered_doc_ids)

        for doc_id in ordered_doc_ids:
            meta = artifacts.doc_store.get(doc_id)
            rows.append(
                {
                    "qid": q.qid,
                    "query": q.text,
                    "doc_id": doc_id,
                    "subreddit": meta.subreddit if meta else "",
                    "title": (meta.title if meta else "").replace("\n", " ").strip(),
                    "excerpt": _excerpt(meta),
                    "label": "",
                }
            )
    return rows
