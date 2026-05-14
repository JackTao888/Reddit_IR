"""CLI entry point for the Flask UI.

Run:
    python -m src.app.cli serve --index-dir data/index --port 5000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..rankers.registry import DEFAULT_RANKER_NAMES
from .app_factory import create_app
from .config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.app.cli",
        description="Run the Reddit IR Flask UI.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Start the development web server.")
    s.add_argument("--index-dir", type=Path, default=Path("data/index"))
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--debug", action="store_true")
    s.add_argument(
        "--default-ranker",
        choices=list(DEFAULT_RANKER_NAMES),
        default="bm25_plain",
    )
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--max-top-k", type=int, default=50)
    s.add_argument(
        "--warmup",
        action="store_true",
        help="Pre-instantiate all rankers at startup (TF-IDF doc-norm precompute).",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "serve":
        cfg = AppConfig(
            index_dir=args.index_dir,
            default_ranker=args.default_ranker,
            default_top_k=args.top_k,
            max_top_k=args.max_top_k,
        )
        app = create_app(cfg)
        if args.warmup:
            app.config["RANKER_POOL"].warmup()
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
