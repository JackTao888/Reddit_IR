"""CLI entry point for evaluation.

Subcommands:
    pool  -- generate pool.csv (label template) from queries + index
    run   -- compute metrics for all rankers using a labeled qrels.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from ..index.store import load_artifacts
from ..rankers.base import BaseRanker
from ..rankers.bm25_ranker import (
    DEFAULT_B,
    DEFAULT_FIELD_WEIGHTS,
    DEFAULT_K1,
    Bm25Ranker,
)
from ..rankers.tfidf_ranker import TfidfRanker
from .config import DEFAULT_POOL_DEPTH, DEFAULT_SHUFFLE_SEED, DEFAULT_TOP_K
from .pool import generate_pool
from .qrels import load_qrels_csv, save_pool_csv
from .queries import load_queries
from .runner import format_summary_table, run_evaluation, write_results_csvs


_VALID_RANKER_NAMES = {"tfidf", "bm25", "bm25_field"}


def _build_default_rankers(artifacts, names: List[str]) -> List[Tuple[str, BaseRanker]]:
    out: List[Tuple[str, BaseRanker]] = []
    for n in names:
        if n == "tfidf":
            out.append((n, TfidfRanker(artifacts)))
        elif n == "bm25":
            out.append((n, Bm25Ranker(artifacts, k1=DEFAULT_K1, b=DEFAULT_B)))
        elif n == "bm25_field":
            out.append(
                (
                    n,
                    Bm25Ranker(
                        artifacts,
                        k1=DEFAULT_K1,
                        b=DEFAULT_B,
                        field_weights=DEFAULT_FIELD_WEIGHTS,
                    ),
                )
            )
        else:
            raise ValueError(f"Unknown ranker: {n}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluate.cli",
        description="Generate label pools and compute IR metrics.",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("pool", help="Generate a labeling pool from queries + index.")
    pp.add_argument(
        "--queries",
        type=Path,
        default=Path("qrel/queries.json"),
        help="Query set JSON (default: qrel/queries.json).",
    )
    pp.add_argument("--index-dir", type=Path, default=Path("data/index"))
    pp.add_argument("--output", type=Path, default=Path("qrel/pool.csv"))
    pp.add_argument("--depth", type=int, default=DEFAULT_POOL_DEPTH)
    pp.add_argument("--seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    pp.add_argument(
        "--rankers",
        nargs="+",
        default=["tfidf", "bm25", "bm25_field"],
        help=f"Rankers to pool from. Choose from {sorted(_VALID_RANKER_NAMES)}.",
    )

    pr = sub.add_parser("run", help="Compute metrics from a labeled qrels CSV.")
    pr.add_argument(
        "--queries",
        type=Path,
        default=Path("qrel/queries.json"),
        help="Query set JSON (default: qrel/queries.json).",
    )
    pr.add_argument("--qrels", type=Path, required=True, help="Labeled pool CSV (e.g. qrel/qrels.csv).")
    pr.add_argument("--index-dir", type=Path, default=Path("data/index"))
    pr.add_argument("--output-dir", type=Path, default=Path("qrel"))
    pr.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    pr.add_argument(
        "--rankers",
        nargs="+",
        default=["tfidf", "bm25", "bm25_field"],
        help=f"Rankers to evaluate. Choose from {sorted(_VALID_RANKER_NAMES)}.",
    )

    return p


def _cmd_pool(args: argparse.Namespace) -> int:
    artifacts = load_artifacts(args.index_dir)
    queries = load_queries(args.queries)
    rankers = _build_default_rankers(artifacts, args.rankers)
    rows = generate_pool(
        artifacts,
        rankers,
        queries,
        pool_depth=args.depth,
        shuffle_seed=args.seed,
    )
    save_pool_csv(rows, args.output)
    print(f"Wrote pool with {len(rows)} rows to {args.output}")
    print("Edit the 'label' column (1 = relevant, 0/blank = not relevant), then save as qrel/qrels.csv and run:")
    print(
        f"  python -m src.evaluate.cli run --queries {args.queries} "
        f"--qrels qrel/qrels.csv --index-dir {args.index_dir}"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    artifacts = load_artifacts(args.index_dir)
    queries = load_queries(args.queries)
    qrels = load_qrels_csv(args.qrels)
    rankers = _build_default_rankers(artifacts, args.rankers)

    results = run_evaluation(rankers, queries, qrels, top_k=args.top_k)
    per_query_path, summary_path = write_results_csvs(results, args.output_dir)

    print()
    print(format_summary_table(results))
    print()
    print(f"Per-query results: {per_query_path}")
    print(f"Summary:           {summary_path}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "pool":
        return _cmd_pool(args)
    if args.command == "run":
        return _cmd_run(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
