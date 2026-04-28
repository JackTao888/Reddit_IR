"""Query set load/save (JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Query:
    qid: str
    text: str
    category: Optional[str] = None


def load_queries(path: Path) -> List[Query]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    queries: List[Query] = []
    for item in data.get("queries", []):
        if not isinstance(item, dict):
            continue
        qid = item.get("qid")
        text = item.get("text")
        if not qid or not text:
            continue
        queries.append(Query(qid=str(qid), text=str(text), category=item.get("category")))
    return queries


def save_queries(queries: Iterable[Query], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"queries": [{k: v for k, v in asdict(q).items() if v is not None} for q in queries]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
