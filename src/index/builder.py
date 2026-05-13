"""Walk preprocessed JSONL and produce a populated ``IndexArtifacts``."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from .config import IndexConfig
from .inverted import InvertedIndex
from .store import DocMetadata, DocStore, IndexArtifacts


log = logging.getLogger(__name__)


_FIELD_TO_TOKENS_KEY = {
    "all": "all_tokens",
    "title": "title_tokens",
    "body": "body_tokens",
    "comments": "comments_tokens",
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


def _input_files(input_dir: Path) -> list:
    return sorted(p for p in input_dir.glob("*.jsonl") if p.is_file())


def _all_tokens_fingerprint(tokens: List[str]) -> bytes:
    """Stable digest of token list (detect duplicate CSV rows / same body)."""
    h = hashlib.blake2b(digest_size=16)
    for w in tokens:
        h.update(w.encode("utf-8", errors="replace"))
        h.update(b"\xff")
    return h.digest()


def build_artifacts(config: IndexConfig) -> IndexArtifacts:
    config.validate()

    indexes = {name: InvertedIndex(name=name) for name in config.fields}
    doc_store = DocStore()
    preprocessing: dict | None = None

    files = _input_files(config.input_dir)
    if not files:
        log.warning("No processed JSONL files in %s", config.input_dir)

    # CSV exports repeat the same thread many times with identical tokens.
    # Keep only the first occurrence per (subreddit, post_id, all_tokens).
    seen_thread_body: set[tuple[str, str, bytes]] = set()
    skipped_duplicate_rows = 0

    for path in files:
        log.info("Indexing %s", path)
        for rec in _iter_jsonl(path):
            doc_id = rec.get("doc_id")
            if not doc_id:
                continue

            sub = (rec.get("subreddit") or "").strip().lower()
            post_id = (rec.get("post_id") or "").strip()
            all_tokens = rec.get("all_tokens", []) or []
            body_key = (sub, post_id, _all_tokens_fingerprint(all_tokens))
            if body_key in seen_thread_body:
                skipped_duplicate_rows += 1
                continue
            seen_thread_body.add(body_key)

            # Capture preprocessing metadata from the first record so the
            # ranker can reproduce query tokenization to match doc tokens.
            if preprocessing is None and isinstance(rec.get("preprocessing"), dict):
                preprocessing = dict(rec["preprocessing"])

            for field_name, idx in indexes.items():
                tokens = rec.get(_FIELD_TO_TOKENS_KEY[field_name], []) or []
                idx.add_document(doc_id, tokens)

            doc_store.add(DocMetadata.from_processed_record(rec))

    artifacts = IndexArtifacts(
        indexes=indexes,
        doc_store=doc_store,
        config=config.to_dict(),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        preprocessing=preprocessing,
    )
    log.info(
        "Index built: docs=%d skipped_duplicate_rows=%d fields=%s preprocessing=%s",
        len(doc_store),
        skipped_duplicate_rows,
        list(indexes.keys()),
        preprocessing,
    )
    return artifacts
