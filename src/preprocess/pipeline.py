"""End-to-end preprocessing pipeline: raw JSONL -> processed JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set

from .config import PreprocessConfig
from .stopwords import build_stopword_set
from .tokens import (
    build_stemmer,
    clean_text,
    normalize_tokens,
    stem_tokens,
    tokenize,
)


log = logging.getLogger(__name__)


def _process_field(
    text: str,
    *,
    cfg: PreprocessConfig,
    stopwords: Set[str],
    stemmer: Callable[[str], str],
) -> List[str]:
    cleaned = clean_text(
        text,
        strip_urls=cfg.strip_urls,
        strip_markdown=cfg.strip_markdown,
        keep_subreddit_refs=cfg.keep_subreddit_refs,
    )
    raw_tokens = tokenize(cleaned)
    tokens = normalize_tokens(
        raw_tokens,
        lowercase=cfg.lowercase,
        min_length=cfg.min_token_length,
        drop_pure_numeric=cfg.drop_pure_numeric,
        stopwords=stopwords if cfg.remove_stopwords else None,
    )
    if cfg.stem:
        tokens = stem_tokens(tokens, stemmer)
    return tokens


def preprocess_doc(
    record: dict,
    *,
    config: PreprocessConfig,
    stopwords: Optional[Set[str]] = None,
    stemmer: Optional[Callable[[str], str]] = None,
) -> dict:
    """Convert one crawler record into a processed record.

    ``stopwords`` and ``stemmer`` are optional so callers can build them once
    and reuse across many docs (the pipeline runner does this).
    """
    cfg = config
    sw = stopwords if stopwords is not None else build_stopword_set(extra_path=cfg.extra_stopwords_path)
    st = stemmer if stemmer is not None else build_stemmer(cfg.stemmer if cfg.stem else "none")

    title = record.get("title", "") or ""
    selftext = record.get("selftext", "") or ""
    comments = record.get("top_comments", []) or []
    comments_blob = " \n ".join(c for c in comments if c)

    title_tokens = _process_field(title, cfg=cfg, stopwords=sw, stemmer=st)
    body_tokens = _process_field(selftext, cfg=cfg, stopwords=sw, stemmer=st)
    comments_tokens = _process_field(comments_blob, cfg=cfg, stopwords=sw, stemmer=st)

    # Empty-doc safety net: keep title tokens even if everything was filtered,
    # so downstream indexing has something to work with.
    if not (title_tokens or body_tokens or comments_tokens) and title:
        title_tokens = normalize_tokens(
            tokenize(title),
            lowercase=cfg.lowercase,
            min_length=1,
            drop_pure_numeric=False,
            stopwords=None,
        )

    all_tokens = title_tokens + body_tokens + comments_tokens

    excerpt = (selftext or "")[: cfg.selftext_excerpt_chars]

    return {
        "doc_id": record.get("doc_id"),
        "post_id": record.get("post_id"),
        "subreddit": record.get("subreddit"),
        "title": title,
        "selftext_excerpt": excerpt,
        "url": record.get("url", ""),
        "permalink": record.get("permalink", ""),
        "score": int(record.get("score", 0) or 0),
        "title_tokens": title_tokens,
        "body_tokens": body_tokens,
        "comments_tokens": comments_tokens,
        "all_tokens": all_tokens,
        "title_len": len(title_tokens),
        "body_len": len(body_tokens),
        "comments_len": len(comments_tokens),
        "doc_len": len(all_tokens),
        "preprocessing": cfg.preprocessing_metadata(),
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _input_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.glob("*.jsonl") if p.is_file())


def run_preprocess(config: PreprocessConfig) -> dict:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sw = build_stopword_set(extra_path=config.extra_stopwords_path)
    stemmer = build_stemmer(config.stemmer if config.stem else "none")

    files = _input_files(config.input_dir)
    if not files:
        log.warning("No input JSONL files in %s", config.input_dir)

    stats: Dict = {
        "total_docs": 0,
        "total_tokens": 0,
        "vocab": set(),
        "per_subreddit": {},
    }

    for in_path in files:
        out_path = config.output_dir / in_path.name
        log.info("Preprocessing %s -> %s", in_path, out_path)

        sub_stats = {"docs": 0, "tokens": 0}
        with open(out_path, "w", encoding="utf-8") as out_f:
            for rec in _iter_jsonl(in_path):
                processed = preprocess_doc(
                    rec, config=config, stopwords=sw, stemmer=stemmer
                )
                out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")

                sub_stats["docs"] += 1
                sub_stats["tokens"] += processed["doc_len"]
                stats["vocab"].update(processed["all_tokens"])

        stats["total_docs"] += sub_stats["docs"]
        stats["total_tokens"] += sub_stats["tokens"]
        stats["per_subreddit"][in_path.stem] = sub_stats

    summary = {
        "total_docs": stats["total_docs"],
        "total_tokens": stats["total_tokens"],
        "vocab_size": len(stats["vocab"]),
        "mean_doc_len": (
            stats["total_tokens"] / stats["total_docs"]
            if stats["total_docs"] > 0
            else 0.0
        ),
        "per_subreddit": stats["per_subreddit"],
    }
    log.info(
        "Preprocess complete: docs=%d tokens=%d vocab~=%d mean_doc_len=%.1f",
        summary["total_docs"],
        summary["total_tokens"],
        summary["vocab_size"],
        summary["mean_doc_len"],
    )
    return summary
