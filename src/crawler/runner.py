"""Crawl orchestration: pacing, retry/backoff, circuit-breaking, manifest."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from .checkpoint import CheckpointStore, Manifest
from .collectors import extract_top_comments, iter_submissions
from .config import CrawlerConfig
from .filters import passes_filters
from .serializers import JsonlWriter, serialize_post

log = logging.getLogger(__name__)


def _classify_error(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "ratelimit" in msg or "429" in msg or "too many" in msg:
        return "rate_limit"
    if "auth" in name or "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
        return "auth"
    if "timeout" in msg or "connection" in msg or "network" in msg:
        return "network"
    if "json" in msg or "parse" in msg:
        return "parse"
    return "unknown"


class Runner:
    def __init__(
        self,
        config: CrawlerConfig,
        *,
        reddit_factory: Optional[Callable[[CrawlerConfig], object]] = None,
        sleeper: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ):
        self.config = config
        self._reddit_factory = reddit_factory
        self._sleep = sleeper
        self._rng = rng or random.Random()
        self.stats: dict = {
            "attempted": 0,
            "saved": 0,
            "skipped_dedup": 0,
            "skipped_filter": 0,
            "errors": {},
            "per_subreddit": {},
        }

    def _record_error(self, category: str) -> None:
        self.stats["errors"][category] = self.stats["errors"].get(category, 0) + 1

    def _wait_between_batches(self) -> None:
        base = self.config.batch_sleep_ms / 1000.0
        jitter = self._rng.uniform(0, self.config.jitter_ms / 1000.0)
        if base + jitter > 0:
            self._sleep(base + jitter)

    def _retry(self, fn: Callable, *, what: str):
        attempts = 0
        delay = 1.0
        while True:
            try:
                return fn()
            except Exception as exc:
                category = _classify_error(exc)
                self._record_error(category)
                attempts += 1
                if attempts > self.config.max_retries:
                    log.error(
                        "Giving up on %s after %d attempts (%s): %s",
                        what,
                        attempts,
                        category,
                        exc,
                    )
                    raise
                jitter = self._rng.uniform(0, 0.5)
                wait = delay + jitter
                log.warning(
                    "Retrying %s (%s) in %.1fs (attempt %d/%d)",
                    what,
                    category,
                    wait,
                    attempts,
                    self.config.max_retries,
                )
                self._sleep(wait)
                delay = min(delay * 2, 30.0)

    def _build_reddit(self):
        if self._reddit_factory is not None:
            return self._reddit_factory(self.config)
        from .client import build_reddit

        return build_reddit(self.config)

    def _process_subreddit(
        self,
        sub: str,
        reddit,
        writer: JsonlWriter,
        checkpoint: CheckpointStore,
    ) -> None:
        cfg = self.config
        per_sub = self.stats["per_subreddit"].setdefault(
            sub,
            {"attempted": 0, "saved": 0, "skipped_dedup": 0, "skipped_filter": 0},
        )
        consecutive_failures = 0

        try:
            listing = iter_submissions(
                reddit,
                sub,
                sort_mode=cfg.sort_mode,
                time_filter=cfg.time_filter,
                limit=cfg.post_limit,
            )
        except Exception as exc:
            self._record_error(_classify_error(exc))
            log.error("Failed to start listing for %s: %s", sub, exc)
            return

        try:
            for submission in listing:
                self.stats["attempted"] += 1
                per_sub["attempted"] += 1

                post_id = getattr(submission, "id", None)
                if not post_id:
                    self._record_error("parse")
                    continue

                if not cfg.refresh and checkpoint.has_seen(sub, post_id):
                    self.stats["skipped_dedup"] += 1
                    per_sub["skipped_dedup"] += 1
                    continue

                if not passes_filters(
                    submission,
                    min_score=cfg.min_score,
                    include_nsfw=cfg.include_nsfw,
                ):
                    self.stats["skipped_filter"] += 1
                    per_sub["skipped_filter"] += 1
                    continue

                try:
                    comments = self._retry(
                        lambda s=submission: extract_top_comments(
                            s, comment_limit=cfg.comment_limit
                        ),
                        what=f"comments[{post_id}]",
                    )
                except Exception:
                    consecutive_failures += 1
                    if consecutive_failures >= cfg.circuit_breaker_threshold:
                        log.error(
                            "Circuit breaker tripped for %s after %d failures; moving on.",
                            sub,
                            consecutive_failures,
                        )
                        break
                    continue
                consecutive_failures = 0

                record = serialize_post(submission, subreddit=sub, top_comments=comments)
                writer.write(sub, record)
                checkpoint.mark_seen(sub, post_id)
                self.stats["saved"] += 1
                per_sub["saved"] += 1

                if self.stats["saved"] % cfg.checkpoint_every == 0:
                    checkpoint.save()

                self._wait_between_batches()
        except Exception as exc:
            self._record_error(_classify_error(exc))
            log.error("Listing iteration failed for %s: %s", sub, exc)

        log.info("Subreddit %s done: %d saved this run.", sub, per_sub["saved"])
        checkpoint.save()

    def run(self) -> dict:
        cfg = self.config
        cfg.validate()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log.info(
            "Starting crawl: subreddits=%s output=%s sort=%s time=%s",
            list(cfg.subreddits),
            cfg.output_dir,
            cfg.sort_mode,
            cfg.time_filter,
        )

        checkpoint = CheckpointStore(cfg.output_dir)
        if cfg.resume:
            checkpoint.load(cfg.subreddits, scan_jsonl=True)
            for sub in cfg.subreddits:
                log.info(
                    "Resume: %s already has %d known posts",
                    sub,
                    len(checkpoint.seen.get(sub, set())),
                )

        reddit = self._build_reddit()

        try:
            with JsonlWriter(cfg.output_dir) as writer:
                for sub in cfg.subreddits:
                    self._process_subreddit(sub, reddit, writer, checkpoint)
        finally:
            checkpoint.save()

        ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        Manifest(cfg.output_dir).append_run(
            params=cfg.to_dict(),
            stats=self.stats,
            started_at=started_at,
            ended_at=ended_at,
        )
        log.info(
            "Crawl complete. saved=%d skipped_dedup=%d skipped_filter=%d errors=%s",
            self.stats["saved"],
            self.stats["skipped_dedup"],
            self.stats["skipped_filter"],
            self.stats["errors"],
        )
        return self.stats
