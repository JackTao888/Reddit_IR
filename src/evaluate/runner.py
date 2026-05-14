"""Run a list of rankers against a labeled qrels file and compute metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ..rankers.base import BaseRanker
from .config import DEFAULT_METRICS_AT, DEFAULT_TOP_K
from .metrics import (
    average_precision,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
)
from .queries import Query


def _retrieve_top_k(ranker: BaseRanker, query: str, top_k: int) -> List[str]:
    return [r.doc_id for r in ranker.search(query, top_k=top_k)]


def run_evaluation(
    rankers: Sequence[Tuple[str, BaseRanker]],
    queries: Iterable[Query],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    top_k: int = DEFAULT_TOP_K,
    metrics_at: Tuple[int, ...] = DEFAULT_METRICS_AT,
) -> dict:
    queries = list(queries)
    fetch_k = max(top_k, max(metrics_at, default=top_k))

    per_ranker: Dict[str, dict] = {}

    for name, ranker in rankers:
        per_query_rows: List[dict] = []
        for q in queries:
            retrieved = _retrieve_top_k(ranker, q.text, fetch_k)
            q_qrels = qrels.get(q.qid, {})

            row = {"ranker": name, "qid": q.qid, "query": q.text}
            for k in metrics_at:
                row[f"P@{k}"] = round(precision_at_k(retrieved, q_qrels, k), 4)
                row[f"NDCG@{k}"] = round(ndcg_at_k(retrieved, q_qrels, k), 4)
            row["AP"] = round(average_precision(retrieved, q_qrels), 4)
            row["n_relevant"] = sum(1 for g in q_qrels.values() if int(g or 0) > 0)
            row["retrieved_at_k"] = len(retrieved[:fetch_k])
            per_query_rows.append(row)

        agg = {"ranker": name, "n_queries": len(queries)}
        for k in metrics_at:
            agg[f"P@{k}"] = round(mean_metric(r[f"P@{k}"] for r in per_query_rows), 4)
            agg[f"NDCG@{k}"] = round(mean_metric(r[f"NDCG@{k}"] for r in per_query_rows), 4)
        # MAP — only over queries that have at least one relevant doc.
        aps_with_rel = [r["AP"] for r in per_query_rows if r["n_relevant"] > 0]
        agg["MAP"] = round(mean_metric(aps_with_rel), 4)

        per_ranker[name] = {"aggregate": agg, "per_query": per_query_rows}

    return {"per_ranker": per_ranker, "metrics_at": metrics_at}


def write_results_csvs(
    results: dict,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_query_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"

    metrics_at = list(results["metrics_at"])
    metric_cols = []
    for k in metrics_at:
        metric_cols.extend([f"P@{k}", f"NDCG@{k}"])

    with open(per_query_path, "w", encoding="utf-8", newline="") as f:
        cols = ["ranker", "qid", "query", *metric_cols, "AP", "n_relevant", "retrieved_at_k"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for ranker_name, payload in results["per_ranker"].items():
            for row in payload["per_query"]:
                writer.writerow({c: row.get(c, "") for c in cols})

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        cols = ["ranker", "n_queries", *metric_cols, "MAP"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for ranker_name, payload in results["per_ranker"].items():
            writer.writerow({c: payload["aggregate"].get(c, "") for c in cols})

    return per_query_path, summary_path


def format_summary_table(results: dict) -> str:
    metrics_at = list(results["metrics_at"])
    metric_cols = []
    for k in metrics_at:
        metric_cols.extend([f"P@{k}", f"NDCG@{k}"])
    cols = ["ranker", "n_queries", *metric_cols, "MAP"]

    rows = [cols]
    for _name, payload in results["per_ranker"].items():
        agg = payload["aggregate"]
        rows.append([str(agg.get(c, "")) for c in cols])

    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    fmt_row = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    return "\n".join([fmt_row(rows[0]), sep, *[fmt_row(r) for r in rows[1:]]])
