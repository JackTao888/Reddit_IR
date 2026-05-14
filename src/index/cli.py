"""CLI entry point for the index builder.

Subcommands:
    build  -- build an index from preprocessed JSONL
    stats  -- print stats from an existing index manifest
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .builder import build_artifacts
from .config import IndexConfig
from .store import load_manifest, save_artifacts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.index.cli",
        description="Build / inspect inverted-index artifacts.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("build", help="Build index from processed JSONL.")
    pb.add_argument("--input", type=Path, default=Path("data/processed"))
    pb.add_argument("--output", type=Path, default=Path("data/index"))

    ps = sub.add_parser("stats", help="Print stats from an existing index.")
    ps.add_argument("--index-dir", type=Path, default=Path("data/index"))

    return p


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = IndexConfig(input_dir=args.input, output_dir=args.output)
    cfg.validate()
    artifacts = build_artifacts(cfg)
    save_artifacts(artifacts, cfg.output_dir)
    print(json.dumps(artifacts.stats(), indent=2))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.index_dir)
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "build":
        return _cmd_build(args)
    if args.command == "stats":
        return _cmd_stats(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
