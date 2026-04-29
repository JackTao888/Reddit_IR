"""Snippet generation with stem-aware query term highlighting.

The index records its preprocessing settings; we reuse them so that a query
like "running" highlights "runs" / "ran" when the index was built with a
stemmer (Snowball "run" matches all of those).
"""

from __future__ import annotations

import html
import re
from typing import List, Optional, Tuple

from markupsafe import Markup

from ..preprocess.tokens import build_stemmer
from ..rankers.query import prepare_query


_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9'_-]*\b")


def _stems_from_query(query: str, preprocessing: dict) -> set:
    return set(prepare_query(query, preprocessing or {}))


def _word_stemmer(preprocessing: dict):
    if preprocessing.get("stem"):
        return build_stemmer(preprocessing.get("stemmer", "snowball"))
    if preprocessing.get("lowercase", True):
        return str.lower
    return lambda w: w


def _find_matches(text: str, target_stems: set, stemmer) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    for m in _WORD_RE.finditer(text):
        word = m.group(0)
        stem = stemmer(word.lower())
        if stem in target_stems:
            matches.append((m.start(), m.end()))
    return matches


def make_snippet(
    text: Optional[str],
    query: str,
    preprocessing: Optional[dict],
    *,
    max_chars: int = 300,
    context_before: int = 60,
) -> Markup:
    """Return safe Markup with `<mark>` around stem-matched query terms."""
    if not text:
        return Markup("")

    cleaned = _WS_RE.sub(" ", text).strip()
    p = preprocessing or {}
    stems = _stems_from_query(query, p)

    if not stems:
        return Markup(html.escape(cleaned[:max_chars] + ("…" if len(cleaned) > max_chars else "")))

    stemmer = _word_stemmer(p)
    matches = _find_matches(cleaned, stems, stemmer)

    if matches:
        first_start = matches[0][0]
        snippet_start = max(0, first_start - context_before)
        snippet_end = min(len(cleaned), snippet_start + max_chars)
    else:
        snippet_start = 0
        snippet_end = min(len(cleaned), max_chars)

    snippet = cleaned[snippet_start:snippet_end]
    local_matches = [
        (s - snippet_start, e - snippet_start)
        for s, e in matches
        if snippet_start <= s and e <= snippet_end
    ]

    out: List[str] = []
    pos = 0
    for s, e in local_matches:
        out.append(html.escape(snippet[pos:s]))
        out.append("<mark>")
        out.append(html.escape(snippet[s:e]))
        out.append("</mark>")
        pos = e
    out.append(html.escape(snippet[pos:]))

    prefix = "…" if snippet_start > 0 else ""
    suffix = "…" if snippet_end < len(cleaned) else ""
    return Markup(prefix + "".join(out) + suffix)
