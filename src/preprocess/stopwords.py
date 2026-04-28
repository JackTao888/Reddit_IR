"""Stopword sets: a robust English baseline + curated Reddit-specific terms.

The English baseline is loaded from NLTK when available, otherwise we use
a small built-in list so the package keeps working in restricted envs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Set


# Used when NLTK's stopwords corpus isn't available. Trimmed list — covers
# the most common English function words. Comparable to NLTK's ~180 entries
# but small enough to ship inline.
_FALLBACK_ENGLISH: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
}


# Reddit-specific terms that appear so often they hurt retrieval signal.
REDDIT_STOPWORDS: Set[str] = {
    "tldr", "tl", "dr", "edit", "edited", "op", "oc", "eli5", "ama",
    "imo", "imho", "fwiw", "iirc", "tbh", "afaik", "ftfy", "iam",
    "lol", "lmao", "rofl", "smh", "tfw", "mfw",
    "upvote", "upvoted", "downvote", "downvoted", "karma",
    "mod", "mods", "moderator", "sub", "subreddit",
    "post", "posted", "comment", "comments", "thread",
    "removed", "deleted", "nsfw", "spoiler",
    "https", "http", "www", "com", "org", "net",
}


def load_english_stopwords() -> Set[str]:
    try:
        from nltk.corpus import stopwords as _sw

        try:
            return set(_sw.words("english"))
        except LookupError:
            return set(_FALLBACK_ENGLISH)
    except ImportError:
        return set(_FALLBACK_ENGLISH)


def load_extra_stopwords(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    out: Set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if token and not token.startswith("#"):
            out.add(token)
    return out


def build_stopword_set(
    *,
    use_english: bool = True,
    use_reddit: bool = True,
    extra: Optional[Iterable[str]] = None,
    extra_path: Optional[Path] = None,
) -> Set[str]:
    out: Set[str] = set()
    if use_english:
        out |= load_english_stopwords()
    if use_reddit:
        out |= REDDIT_STOPWORDS
    if extra:
        out |= {w.lower() for w in extra}
    if extra_path is not None:
        out |= load_extra_stopwords(extra_path)
    return out
