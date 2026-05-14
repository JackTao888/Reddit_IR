"""Query preparation that mirrors the M2 preprocessing pipeline.

Reads preprocessing settings from the index artifact's ``preprocessing`` dict
so query tokens always match the way documents were tokenized.
"""

from __future__ import annotations

from typing import List, Optional

from ..preprocess.stopwords import build_stopword_set
from ..preprocess.tokens import (
    build_stemmer,
    clean_text,
    normalize_tokens,
    stem_tokens,
    tokenize,
)


_DEFAULTS = {
    "lowercase": True,
    "remove_stopwords": True,
    "stem": True,
    "stemmer": "snowball",
    "min_token_length": 2,
    "drop_pure_numeric": True,
    "strip_urls": True,
    "strip_markdown": True,
    "keep_subreddit_refs": False,
}


def prepare_query(query: str, preprocessing: Optional[dict] = None) -> List[str]:
    p = {**_DEFAULTS, **(preprocessing or {})}

    cleaned = clean_text(
        query,
        strip_urls=p["strip_urls"],
        strip_markdown=p["strip_markdown"],
        keep_subreddit_refs=p["keep_subreddit_refs"],
    )
    raw_tokens = tokenize(cleaned)

    sw = build_stopword_set() if p["remove_stopwords"] else set()
    tokens = normalize_tokens(
        raw_tokens,
        lowercase=p["lowercase"],
        min_length=p["min_token_length"],
        drop_pure_numeric=p["drop_pure_numeric"],
        stopwords=sw if p["remove_stopwords"] else None,
    )

    if p["stem"] and p["stemmer"] != "none":
        stemmer = build_stemmer(p["stemmer"])
        tokens = stem_tokens(tokens, stemmer)

    return tokens
