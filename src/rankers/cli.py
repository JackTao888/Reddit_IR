"""CLI entry point for ad-hoc search against a built index.

Examples:
    python -m src.rankers.cli search --query "python async" --ranker bm25_plain
    python -m src.rankers.cli search --ranker bm25_field
    python -m src.rankers.cli search --query "..." --ranker tfidf_cosine --top-k 5
    python -m src.rankers.cli search --query "..." --ranker twostage_bm25_tfidf --stage1-k 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..index.store import load_artifacts
from .bm25_ranker import DEFAULT_B, DEFAULT_FIELD_WEIGHTS, DEFAULT_K1
from .registry import CLI_RANKER_CHOICES, build_ranker, canonical_ranker_name
from .two_stage_ranker import TwoStageBm25TfidfRanker


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
    s.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query string. Omit to read interactively from stdin.",
    )
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument(
        "--ranker",
        choices=CLI_RANKER_CHOICES,
        default="bm25_plain",
        help="Ranker id. Legacy aliases: tfidf→tfidf_cosine, bm25→bm25_plain.",
    )
    s.add_argument("--k1", type=float, default=DEFAULT_K1)
    s.add_argument("--b", type=float, default=DEFAULT_B)
    s.add_argument(
        "--field-aware",
        action="store_true",
        help="Shorthand: same as --ranker bm25_field (overrides --ranker when set).",
    )
    s.add_argument("--w-title", type=float, default=DEFAULT_FIELD_WEIGHTS["title"])
    s.add_argument("--w-body", type=float, default=DEFAULT_FIELD_WEIGHTS["body"])
    s.add_argument(
        "--w-comments",
        type=float,
        default=DEFAULT_FIELD_WEIGHTS["comments"],
    )
    s.add_argument(
        "--stage1-k",
        type=int,
        default=TwoStageBm25TfidfRanker.DEFAULT_STAGE1_K,
        help="BM25 candidate pool size for twostage_bm25_tfidf (default: %(default)s).",
    )
    s.add_argument(
        "--lsi-k",
        type=int,
        default=100,
        help="Truncated SVD latent dimensions for ranker=lsi (default: %(default)s).",
    )
    s.add_argument(
        "--lsi-max-vocab",
        type=int,
        default=8000,
        help="Max vocabulary columns for LSI matrix (default: %(default)s).",
    )
    s.add_argument(
        "--prf-depth",
        type=int,
        default=5,
        help="First-pass BM25 depth for bm25_prf (default: %(default)s).",
    )
    s.add_argument(
        "--prf-expand",
        type=int,
        default=15,
        help="Number of expansion terms for bm25_prf (default: %(default)s).",
    )
    return p


def _make_ranker(args: argparse.Namespace, artifacts):
    name = canonical_ranker_name(args.ranker)
    if args.field_aware and name == "bm25_plain":
        name = "bm25_field"
    field_weights = None
    if name == "bm25_field":
        field_weights = {
            "title": args.w_title,
            "body": args.w_body,
            "comments": args.w_comments,
        }
    return build_ranker(
        name,
        artifacts,
        k1=args.k1,
        b=args.b,
        field_weights=field_weights,
        stage1_k=args.stage1_k,
        lsi_k=args.lsi_k,
        lsi_max_vocab=args.lsi_max_vocab,
        prf_depth=args.prf_depth,
        prf_expansion_terms=args.prf_expand,
    )


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
    print(
        f"Loaded index: n_docs={len(artifacts.doc_store)} ranker={ranker.name}",
        file=sys.stderr,
    )

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
