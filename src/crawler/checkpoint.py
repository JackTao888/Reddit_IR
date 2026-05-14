"""Resume state and run manifests.

``CheckpointStore`` tracks ``post_id``s already saved per subreddit so reruns
are idempotent. ``Manifest`` records human-readable run summaries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Set

from .serializers import iter_jsonl_records


class CheckpointStore:
    FILENAME = ".checkpoint.json"

    def __init__(self, output_dir: Path):
        self.path = Path(output_dir) / self.FILENAME
        self.seen: Dict[str, Set[str]] = {}

    def load(self, subreddits: Iterable[str], *, scan_jsonl: bool = True) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for sub, ids in (data.get("seen_post_ids") or {}).items():
                    self.seen[sub] = set(ids)
            except json.JSONDecodeError:
                self.seen = {}

        # Backstop: rebuild from existing JSONL files even if checkpoint file is
        # missing/corrupted. This keeps reruns safe across machine moves.
        if scan_jsonl:
            for sub in subreddits:
                jsonl = self.path.parent / f"{sub.lower()}.jsonl"
                if not jsonl.exists():
                    continue
                self.seen.setdefault(sub, set())
                for rec in iter_jsonl_records(jsonl):
                    pid = rec.get("post_id")
                    if pid:
                        self.seen[sub].add(pid)

    def has_seen(self, subreddit: str, post_id: str) -> bool:
        return post_id in self.seen.get(subreddit, set())

    def mark_seen(self, subreddit: str, post_id: str) -> None:
        self.seen.setdefault(subreddit, set()).add(post_id)

    def save(self) -> None:
        payload = {
            "seen_post_ids": {sub: sorted(ids) for sub, ids in self.seen.items()},
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class Manifest:
    FILENAME = "crawl_manifest.json"

    def __init__(self, output_dir: Path):
        self.path = Path(output_dir) / self.FILENAME

    def append_run(
        self,
        *,
        params: dict,
        stats: dict,
        started_at: str,
        ended_at: str,
    ) -> None:
        existing = {"runs": []}
        if self.path.exists():
            try:
                parsed = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict) and isinstance(parsed.get("runs"), list):
                    existing = parsed
            except json.JSONDecodeError:
                pass
        existing["runs"].append(
            {
                "started_at": started_at,
                "ended_at": ended_at,
                "params": params,
                "stats": stats,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
