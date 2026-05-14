"""Command-line entry point for the crawler.

Run:
    python -m src.crawler.cli --subreddits science programming cooking \\
        --limit 1000 --comments 5 --sort top --time month
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    DEFAULT_BATCH_SLEEP_MS,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_COMMENT_LIMIT,
    DEFAULT_JITTER_MS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_POST_LIMIT,
    DEFAULT_USER_AGENT,
    VALID_SORTS,
    VALID_TIME_FILTERS,
    CrawlerConfig,
    load_env_credentials,
)
from .runner import Runner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.crawler.cli",
        description="Reddit crawler for the JHU 666 IR final project.",
    )
    p.add_argument(
        "--subreddits",
        nargs="+",
        required=True,
        help="Subreddits to crawl (without the r/ prefix).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for JSONL files (default: %(default)s).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_POST_LIMIT,
        help="Posts to pull per subreddit (default: %(default)s, max ~1000 from Reddit).",
    )
    p.add_argument(
        "--comments",
        type=int,
        default=DEFAULT_COMMENT_LIMIT,
        help="Top comments to keep per post (default: %(default)s).",
    )
    p.add_argument("--sort", choices=sorted(VALID_SORTS), default="top")
    p.add_argument(
        "--time",
        dest="time_filter",
        choices=sorted(VALID_TIME_FILTERS),
        default="month",
    )
    p.add_argument("--min-score", type=int, default=None)
    p.add_argument("--include-nsfw", action="store_true")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from prior checkpoints.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch records even if already saved.",
    )
    p.add_argument("--batch-sleep-ms", type=int, default=DEFAULT_BATCH_SLEEP_MS)
    p.add_argument("--jitter-ms", type=int, default=DEFAULT_JITTER_MS)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
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

    cfg = CrawlerConfig(
        subreddits=args.subreddits,
        output_dir=args.output,
        sort_mode=args.sort,
        time_filter=args.time_filter,
        post_limit=args.limit,
        comment_limit=args.comments,
        min_score=args.min_score,
        include_nsfw=args.include_nsfw,
        resume=not args.no_resume,
        refresh=args.refresh,
        batch_sleep_ms=args.batch_sleep_ms,
        jitter_ms=args.jitter_ms,
        max_retries=args.max_retries,
        checkpoint_every=args.checkpoint_every,
        user_agent=args.user_agent,
    )
    cfg.validate()
    load_env_credentials(cfg)

    if not cfg.client_id or not cfg.client_secret:
        print(
            "ERROR: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set "
            "(see .env.example).",
            file=sys.stderr,
        )
        return 2

    Runner(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
