"""Evaluation harness for the JHU 666 IR final project.

Public API:
- Query, load_queries, save_queries
- Qrels: pool/qrels CSV I/O (load_qrels_csv, save_pool_csv)
- metrics: precision_at_k, average_precision, ndcg_at_k, mean_*
- generate_pool: union top-N results across rankers, shuffled
- run_evaluation: per-query + aggregate metrics across rankers
- main: CLI entry point (``python -m src.evaluate.cli``)
"""

from .metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
)
from .pool import generate_pool
from .qrels import load_qrels_csv, save_pool_csv
from .queries import Query, load_queries, save_queries
from .runner import run_evaluation

__all__ = [
    "Query",
    "load_queries",
    "save_queries",
    "load_qrels_csv",
    "save_pool_csv",
    "precision_at_k",
    "average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "mean_average_precision",
    "mean_metric",
    "generate_pool",
    "run_evaluation",
]
