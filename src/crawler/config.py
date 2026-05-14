"""Typed configuration for the crawler.

All tunable parameters live here so other modules stay parameter-free.
Credentials are loaded from environment variables (or a ``.env`` file)
via :func:`load_env_credentials` and never persisted alongside the
serialized config (see :meth:`CrawlerConfig.to_dict`).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_USER_AGENT = "JHU666RedditIR/1.0 (educational project)"

# Reddit OAuth allows ~60 requests/minute; aim for ~55 with jitter to
# leave headroom for paginated calls inside PRAW.
DEFAULT_BATCH_SLEEP_MS = 1100
DEFAULT_JITTER_MS = 250
DEFAULT_MAX_RETRIES = 3
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_CIRCUIT_BREAKER = 5
DEFAULT_RATELIMIT_SECONDS = 300
DEFAULT_POST_LIMIT = 1000
DEFAULT_COMMENT_LIMIT = 5

VALID_SORTS = {"hot", "new", "top", "rising"}
VALID_TIME_FILTERS = {"all", "day", "hour", "month", "week", "year"}


@dataclass
class CrawlerConfig:
    subreddits: Sequence[str]
    output_dir: Path

    sort_mode: str = "top"
    time_filter: str = "month"
    post_limit: int = DEFAULT_POST_LIMIT
    comment_limit: int = DEFAULT_COMMENT_LIMIT

    min_score: Optional[int] = None
    include_nsfw: bool = False

    resume: bool = True
    refresh: bool = False

    batch_sleep_ms: int = DEFAULT_BATCH_SLEEP_MS
    jitter_ms: int = DEFAULT_JITTER_MS
    max_retries: int = DEFAULT_MAX_RETRIES
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER
    ratelimit_seconds: int = DEFAULT_RATELIMIT_SECONDS

    client_id: str = ""
    client_secret: str = ""
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        self.subreddits = [s.strip() for s in self.subreddits if s and s.strip()]
        self.output_dir = Path(self.output_dir)

    def validate(self) -> None:
        if not self.subreddits:
            raise ValueError("At least one subreddit is required")
        if self.sort_mode not in VALID_SORTS:
            raise ValueError(f"sort_mode must be one of {sorted(VALID_SORTS)}")
        if self.time_filter not in VALID_TIME_FILTERS:
            raise ValueError(f"time_filter must be one of {sorted(VALID_TIME_FILTERS)}")
        if self.post_limit <= 0:
            raise ValueError("post_limit must be positive")
        if self.comment_limit < 0:
            raise ValueError("comment_limit cannot be negative")
        if self.batch_sleep_ms < 0 or self.jitter_ms < 0:
            raise ValueError("sleep/jitter must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.circuit_breaker_threshold <= 0:
            raise ValueError("circuit_breaker_threshold must be > 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d.pop("client_id", None)
        d.pop("client_secret", None)
        return d


def load_env_credentials(config: CrawlerConfig) -> None:
    """Populate ``client_id``, ``client_secret`` and ``user_agent`` from the
    environment. Called explicitly so unit tests can avoid touching env state.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    config.client_id = os.environ.get("REDDIT_CLIENT_ID", config.client_id)
    config.client_secret = os.environ.get("REDDIT_CLIENT_SECRET", config.client_secret)
    config.user_agent = os.environ.get("REDDIT_USER_AGENT", config.user_agent)
