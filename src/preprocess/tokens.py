"""Text cleaning, tokenization, and stemming helpers.

NLTK is a *soft* dependency: when available we use ``word_tokenize`` and a
proper stemmer; otherwise we fall back to a regex tokenizer and identity
stemmer so the package is importable in restricted environments.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Optional, Set


# ---------- text cleaning ----------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SUB_REF_RE = re.compile(r"/?r/[A-Za-z0-9_]+", re.IGNORECASE)
_USER_REF_RE = re.compile(r"/?u/[A-Za-z0-9_-]+", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_DECOR_RE = re.compile(r"[*_~>#]+")
_DELETED_RE = re.compile(r"\[(removed|deleted)\]", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def clean_text(
    text: str,
    *,
    strip_urls: bool = True,
    strip_markdown: bool = True,
    keep_subreddit_refs: bool = False,
) -> str:
    """Apply Reddit-aware text cleaning before tokenization."""
    if not text:
        return ""
    s = text

    if strip_markdown:
        s = _CODE_FENCE_RE.sub(" ", s)
        s = _INLINE_CODE_RE.sub(" ", s)
        s = _MD_LINK_RE.sub(r"\1", s)

    if strip_urls:
        s = _URL_RE.sub(" ", s)

    if not keep_subreddit_refs:
        s = _SUB_REF_RE.sub(" ", s)
        s = _USER_REF_RE.sub(" ", s)

    s = _DELETED_RE.sub(" ", s)

    if strip_markdown:
        s = _MD_DECOR_RE.sub(" ", s)

    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------- tokenization ----------

_REGEX_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\d+")


def _regex_tokenize(text: str) -> List[str]:
    return _REGEX_TOKEN_RE.findall(text)


def _nltk_tokenize_or_none(text: str) -> Optional[List[str]]:
    try:
        from nltk.tokenize import word_tokenize  # type: ignore

        try:
            return word_tokenize(text)
        except LookupError:
            return None
    except ImportError:
        return None


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    nltk_tokens = _nltk_tokenize_or_none(text)
    if nltk_tokens is not None:
        return nltk_tokens
    return _regex_tokenize(text)


# ---------- post-tokenization filtering ----------

_PURE_NUMERIC_RE = re.compile(r"^\d+$")


def normalize_tokens(
    tokens: Iterable[str],
    *,
    lowercase: bool = True,
    min_length: int = 2,
    drop_pure_numeric: bool = True,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    out: List[str] = []
    sw = stopwords or set()
    for raw in tokens:
        tok = raw.lower() if lowercase else raw
        # NLTK's word_tokenize emits punctuation/contraction markers — drop them.
        if not tok or not any(ch.isalnum() for ch in tok):
            continue
        if len(tok) < min_length:
            continue
        if drop_pure_numeric and _PURE_NUMERIC_RE.match(tok):
            continue
        if tok in sw:
            continue
        out.append(tok)
    return out


# ---------- stemming ----------

def build_stemmer(name: str) -> Callable[[str], str]:
    """Return a stemmer function for the given name.

    Falls back to identity if the requested stemmer or NLTK isn't available.
    """
    name = (name or "none").lower()
    if name == "none":
        return lambda t: t
    try:
        if name == "porter":
            from nltk.stem.porter import PorterStemmer  # type: ignore

            stemmer = PorterStemmer()
            return stemmer.stem
        if name == "snowball":
            from nltk.stem.snowball import SnowballStemmer  # type: ignore

            stemmer = SnowballStemmer("english")
            return stemmer.stem
    except ImportError:
        pass
    return lambda t: t


def stem_tokens(tokens: Iterable[str], stemmer: Callable[[str], str]) -> List[str]:
    return [stemmer(t) for t in tokens]
