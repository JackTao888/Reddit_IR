"""CLI entry point for the preprocessor.

Run:
    python -m src.preprocess.cli --input data/raw --output data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import VALID_STEMMERS, PreprocessConfig
from .pipeline import run_preprocess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.preprocess.cli",
        description="Preprocess crawler JSONL into tokenized JSONL for indexing.",
    )
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--no-lowercase", action="store_true")
    p.add_argument("--no-stopwords", action="store_true")
    p.add_argument("--no-stem", action="store_true")
    p.add_argument(
        "--stemmer",
        choices=sorted(VALID_STEMMERS),
        default="snowball",
    )
    p.add_argument("--min-token-len", type=int, default=2)
    p.add_argument("--keep-numeric", action="store_true")
    p.add_argument("--keep-urls", action="store_true")
    p.add_argument("--keep-markdown", action="store_true")
    p.add_argument("--keep-subreddit-refs", action="store_true")
    p.add_argument("--excerpt-chars", type=int, default=500)
    p.add_argument(
        "--extra-stopwords",
        type=Path,
        default=None,
        help="Path to a plain-text file with one extra stopword per line.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = PreprocessConfig(
        input_dir=args.input,
        output_dir=args.output,
        lowercase=not args.no_lowercase,
        remove_stopwords=not args.no_stopwords,
        stem=not args.no_stem,
        stemmer=args.stemmer,
        min_token_length=args.min_token_len,
        drop_pure_numeric=not args.keep_numeric,
        strip_urls=not args.keep_urls,
        strip_markdown=not args.keep_markdown,
        keep_subreddit_refs=args.keep_subreddit_refs,
        selftext_excerpt_chars=args.excerpt_chars,
        extra_stopwords_path=args.extra_stopwords,
    )
    cfg.validate()

    summary = run_preprocess(cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
