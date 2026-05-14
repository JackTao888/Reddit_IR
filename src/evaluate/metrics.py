"""IR metrics: P@k, AP/MAP, DCG/NDCG.

All functions accept:
- ``retrieved``: ordered list of doc_ids (top-1 first)
- ``qrels``: ``{doc_id: grade}`` (any grade > 0 counts as relevant for P/AP;
  NDCG uses the actual grade)

Unjudged docs (not in ``qrels``) are treated as grade 0, matching the TREC
convention used with pooled labeling.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Mapping


def _rel(qrels: Mapping[str, int], doc_id: str) -> int:
    return int(qrels.get(doc_id, 0) or 0)


def precision_at_k(retrieved: List[str], qrels: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if _rel(qrels, d) > 0)
    return hits / k


def recall_at_k(retrieved: List[str], qrels: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    total_rel = sum(1 for g in qrels.values() if int(g or 0) > 0)
    if total_rel == 0:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if _rel(qrels, d) > 0)
    return hits / total_rel


def average_precision(retrieved: List[str], qrels: Mapping[str, int]) -> float:
    total_rel = sum(1 for g in qrels.values() if int(g or 0) > 0)
    if total_rel == 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(retrieved, 1):
        if _rel(qrels, doc_id) > 0:
            hits += 1
            sum_prec += hits / i
    return sum_prec / total_rel


def dcg_at_k(retrieved: List[str], qrels: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(
        _rel(qrels, doc_id) / math.log2(i + 1)
        for i, doc_id in enumerate(retrieved[:k], 1)
    )


def ndcg_at_k(retrieved: List[str], qrels: Mapping[str, int], k: int) -> float:
    if k <= 0 or not qrels:
        return 0.0
    dcg = dcg_at_k(retrieved, qrels, k)
    grades_desc = sorted((int(g or 0) for g in qrels.values() if int(g or 0) > 0), reverse=True)
    grades_desc = grades_desc[:k]
    if not grades_desc:
        return 0.0
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(grades_desc, 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_metric(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0


def mean_average_precision(
    retrieved_per_query: Mapping[str, List[str]],
    qrels_per_query: Mapping[str, Mapping[str, int]],
) -> float:
    aps: List[float] = []
    for qid, q_qrels in qrels_per_query.items():
        if not q_qrels:
            continue
        retrieved = retrieved_per_query.get(qid, [])
        aps.append(average_precision(retrieved, q_qrels))
    return mean_metric(aps)
