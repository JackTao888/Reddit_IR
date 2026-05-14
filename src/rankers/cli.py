"""CLI entry point for ad-hoc search against a built index.

Examples:
    python -m src.rankers.cli search --query "python async" --ranker bm25
    python -m src.rankers.cli search --ranker bm25 --field-aware
    python -m src.rankers.cli search --query "..." --ranker tfidf --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..index.store import load_artifacts
from .bm25_ranker import DEFAULT_B, DEFAULT_FIELD_WEIGHTS, DEFAULT_K1, Bm25Ranker
from .tfidf_ranker import TfidfRanker


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.rankers.cli",
        description="Search an indexed Reddit corpus.",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="Search using a chosen ranker.")
    s.add_argument("--index-dir", type=Path, default=Path("data/index"))
    s.add_argument("--query", type=str, default=None,
                   help="Query string. Omit to read interactively from stdin.")
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--ranker", choices=["tfidf", "bm25"], default="bm25")

    s.add_argument("--k1", type=float, default=DEFAULT_K1)
    s.add_argument("--b", type=float, default=DEFAULT_B)
    s.add_argument("--field-aware", action="store_true",
                   help="(BM25 only) score per-field and combine.")
    s.add_argument("--w-title", type=float, default=DEFAULT_FIELD_WEIGHTS["title"])
    s.add_argument("--w-body", type=float, default=DEFAULT_FIELD_WEIGHTS["body"])
    s.add_argument("--w-comments", type=float, default=DEFAULT_FIELD_WEIGHTS["comments"])
    return p


def _make_ranker(args: argparse.Namespace, artifacts):
    if args.ranker == "tfidf":
        return TfidfRanker(artifacts)
    field_weights = None
    if args.field_aware:
        field_weights = {
            "title": args.w_title,
            "body": args.w_body,
            "comments": args.w_comments,
        }
    return Bm25Ranker(artifacts, k1=args.k1, b=args.b, field_weights=field_weights)


def _print_results(results, artifacts, query: str) -> None:
    if not results:
        print(f"(no results for: {query!r})")
        return
    print(f"\nQuery: {query}")
    for r in results:
        meta = artifacts.doc_store.get(r.doc_id)
        title = meta.title if meta else "(unknown)"
        sub = meta.subreddit if meta else "?"
        print(f"  {r.rank:2d}. [{sub}] (score={r.score:.4f}) {title}")
        if meta and meta.permalink:
            print(f"      https://reddit.com{meta.permalink}")


def _cmd_search(args: argparse.Namespace) -> int:
    artifacts = load_artifacts(args.index_dir)
    ranker = _make_ranker(args, artifacts)
    print(f"Loaded index: n_docs={len(artifacts.doc_store)} ranker={ranker.name}",
          file=sys.stderr)

    if args.query is not None:
        results = ranker.search(args.query, top_k=args.top_k)
        _print_results(results, artifacts, args.query)
        return 0

    print("Enter queries (blank line or Ctrl-D to exit):", file=sys.stderr)
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                print()
                break
            if not line:
                break
            results = ranker.search(line, top_k=args.top_k)
            _print_results(results, artifacts, line)
    except KeyboardInterrupt:
        print()
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.command == "search":
        return _cmd_search(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
