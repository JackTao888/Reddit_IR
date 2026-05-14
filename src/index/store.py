"""Document metadata store + persistence for index artifacts."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .inverted import InvertedIndex


ARTIFACT_FILENAME = "index.pkl"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class DocMetadata:
    doc_id: str
    post_id: str
    subreddit: str
    title: str
    selftext_excerpt: str
    url: str
    permalink: str
    score: int
    title_len: int
    body_len: int
    comments_len: int
    doc_len: int

    @classmethod
    def from_processed_record(cls, rec: dict) -> "DocMetadata":
        return cls(
            doc_id=rec["doc_id"],
            post_id=rec.get("post_id", ""),
            subreddit=rec.get("subreddit", ""),
            title=rec.get("title", ""),
            selftext_excerpt=rec.get("selftext_excerpt", ""),
            url=rec.get("url", ""),
            permalink=rec.get("permalink", ""),
            score=int(rec.get("score", 0) or 0),
            title_len=int(rec.get("title_len", 0) or 0),
            body_len=int(rec.get("body_len", 0) or 0),
            comments_len=int(rec.get("comments_len", 0) or 0),
            doc_len=int(rec.get("doc_len", 0) or 0),
        )


@dataclass
class DocStore:
    docs: Dict[str, DocMetadata] = field(default_factory=dict)

    def add(self, meta: DocMetadata) -> None:
        self.docs[meta.doc_id] = meta

    def get(self, doc_id: str) -> Optional[DocMetadata]:
        return self.docs.get(doc_id)

    def all_doc_ids(self) -> List[str]:
        return list(self.docs.keys())

    def __len__(self) -> int:
        return len(self.docs)


@dataclass
class IndexArtifacts:
    indexes: Dict[str, InvertedIndex]
    doc_store: DocStore
    config: dict
    built_at: str
    preprocessing: Optional[dict] = None

    @property
    def all_index(self) -> InvertedIndex:
        return self.indexes["all"]

    def stats(self) -> dict:
        return {
            "built_at": self.built_at,
            "n_docs": len(self.doc_store),
            "indexes": {name: idx.stats() for name, idx in self.indexes.items()},
            "preprocessing": self.preprocessing,
        }


def save_artifacts(artifacts: IndexArtifacts, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / ARTIFACT_FILENAME, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        **artifacts.stats(),
        "config": artifacts.config,
        "artifact_file": ARTIFACT_FILENAME,
    }
    with open(output_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_artifacts(output_dir: Path) -> IndexArtifacts:
    output_dir = Path(output_dir)
    artifact_path = output_dir / ARTIFACT_FILENAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"No index artifact at {artifact_path}")
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def load_manifest(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    path = output_dir / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
