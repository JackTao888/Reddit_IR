"""PRAW client construction. Kept as the only place that touches PRAW
directly so the rest of the package is testable without network access.
"""

from __future__ import annotations

import logging

from .config import CrawlerConfig

log = logging.getLogger(__name__)


def build_reddit(config: CrawlerConfig):
    if not config.client_id or not config.client_secret:
        raise RuntimeError(
            "Reddit API credentials missing. "
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your environment "
            "or .env file (see .env.example)."
        )

    try:
        import praw
    except ImportError as e:
        raise RuntimeError(
            "PRAW is not installed. Install with: pip install -r requirements.txt"
        ) from e

    reddit = praw.Reddit(
        client_id=config.client_id,
        client_secret=config.client_secret,
        user_agent=config.user_agent,
        ratelimit_seconds=config.ratelimit_seconds,
        check_for_async=False,
    )
    reddit.read_only = True
    log.info(
        "PRAW client initialized (read_only=%s, ratelimit_seconds=%d)",
        reddit.read_only,
        config.ratelimit_seconds,
    )
    return reddit
