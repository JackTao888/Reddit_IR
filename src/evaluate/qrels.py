"""Pool / qrels CSV I/O.

The pool CSV and the qrels CSV share the exact same schema:

    qid, query, doc_id, subreddit, title, excerpt, label

The only difference is whether the ``label`` column is populated. Empty or
non-integer labels are interpreted as ``0`` (not relevant). Graded relevance
is supported by writing integers > 1.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


POOL_COLUMNS: List[str] = [
    "qid",
    "query",
    "doc_id",
    "subreddit",
    "title",
    "excerpt",
    "label",
]


def save_pool_csv(rows: Iterable[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POOL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in POOL_COLUMNS})


def _parse_label(value) -> int:
    if value is None:
        return 0
    s = str(value).strip()
    if not s:
        return 0
    try:
        n = int(s)
    except ValueError:
        try:
            n = int(float(s))
        except ValueError:
            return 0
    return max(0, n)


def load_qrels_csv(path: Path) -> Dict[str, Dict[str, int]]:
    """Load labeled pool/qrels CSV into ``{qid: {doc_id: grade}}``."""
    path = Path(path)
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("qid") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            if not qid or not doc_id:
                continue
            grade = _parse_label(row.get("label"))
            qrels.setdefault(qid, {})[doc_id] = grade
    return qrels
