"""Content quality filters applied before serialization."""

from __future__ import annotations

from typing import Optional


_DELETED_BODIES = {"[removed]", "[deleted]", "[ Removed by Reddit ]".lower()}


def is_removed_or_deleted(submission) -> bool:
    title = (getattr(submission, "title", "") or "").strip()
    body = (getattr(submission, "selftext", "") or "").strip().lower()
    if not title:
        return True
    if body in _DELETED_BODIES and not title:
        return True
    return False


def passes_filters(
    submission,
    *,
    min_score: Optional[int],
    include_nsfw: bool,
) -> bool:
    if is_removed_or_deleted(submission):
        return False
    if not include_nsfw and bool(getattr(submission, "over_18", False)):
        return False
    if min_score is not None and int(getattr(submission, "score", 0) or 0) < min_score:
        return False
    return True
