"""Thin wrappers over PRAW listing/comment APIs.

Kept duck-typed so ``Runner`` can be unit-tested with simple fakes that
mimic ``praw.models.Submission`` and ``praw.models.CommentForest``.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

log = logging.getLogger(__name__)


def iter_submissions(reddit, subreddit: str, *, sort_mode: str, time_filter: str, limit: int):
    sub = reddit.subreddit(subreddit)
    if sort_mode == "top":
        return sub.top(time_filter=time_filter, limit=limit)
    if sort_mode == "hot":
        return sub.hot(limit=limit)
    if sort_mode == "new":
        return sub.new(limit=limit)
    if sort_mode == "rising":
        return sub.rising(limit=limit)
    raise ValueError(f"Unsupported sort_mode: {sort_mode}")


def extract_top_comments(submission, *, comment_limit: int) -> List[str]:
    if comment_limit <= 0:
        return []

    bodies: List[str] = []
    forest = getattr(submission, "comments", None)
    if forest is None:
        return bodies

    try:
        replace_more = getattr(forest, "replace_more", None)
        if callable(replace_more):
            try:
                replace_more(limit=0)
            except Exception as e:  # pragma: no cover - network-side
                log.debug(
                    "replace_more failed for %s: %s",
                    getattr(submission, "id", "?"),
                    e,
                )

        iterable: Iterable = forest
        for comment in iterable:
            body = getattr(comment, "body", None)
            if not body:
                continue
            if body in ("[removed]", "[deleted]"):
                continue
            bodies.append(body)
            if len(bodies) >= comment_limit:
                break
    except Exception as e:  # pragma: no cover - defensive
        log.warning(
            "Failed extracting comments for %s: %s",
            getattr(submission, "id", "?"),
            e,
        )

    return bodies
