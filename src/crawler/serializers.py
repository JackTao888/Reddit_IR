"""Schema enforcement and JSONL writing.

Each crawled subreddit gets its own ``<subreddit>.jsonl`` file in the
configured output directory. Records follow the schema documented in
``plan.md`` (see M1.1).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional


_WS_RE = re.compile(r"\s+")


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return _WS_RE.sub(" ", value).strip()


def build_doc_id(subreddit: str, post_id: str) -> str:
    return f"{subreddit.lower()}_{post_id}"


def serialize_post(submission, *, subreddit: str, top_comments: List[str]) -> dict:
    return {
        "doc_id": build_doc_id(subreddit, submission.id),
        "post_id": submission.id,
        "subreddit": subreddit,
        "title": _normalize_text(getattr(submission, "title", "")),
        "selftext": _normalize_text(getattr(submission, "selftext", "")),
        "top_comments": [_normalize_text(c) for c in top_comments if c],
        "url": getattr(submission, "url", "") or "",
        "permalink": getattr(submission, "permalink", "") or "",
        "created_utc": float(getattr(submission, "created_utc", 0.0) or 0.0),
        "score": int(getattr(submission, "score", 0) or 0),
        "num_comments": int(getattr(submission, "num_comments", 0) or 0),
        "over_18": bool(getattr(submission, "over_18", False)),
        "is_self": bool(getattr(submission, "is_self", False)),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class JsonlWriter:
    """Append-only JSONL writer, one open file per subreddit."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles: Dict[str, IO[str]] = {}

    def path_for(self, subreddit: str) -> Path:
        return self.output_dir / f"{subreddit.lower()}.jsonl"

    def write(self, subreddit: str, record: dict) -> None:
        fh = self._handles.get(subreddit)
        if fh is None:
            fh = open(self.path_for(subreddit), "a", encoding="utf-8")
            self._handles[subreddit] = fh
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()

    def close(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._handles.clear()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def iter_jsonl_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
