"""Tests for the inverted-index package.

These don't depend on PRAW or NLTK — synthetic processed records exercise
the builder, persistence, and reload paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.index.builder import build_artifacts
from src.index.config import IndexConfig
from src.index.inverted import InvertedIndex
from src.index.store import (
    DocMetadata,
    DocStore,
    IndexArtifacts,
    load_artifacts,
    load_manifest,
    save_artifacts,
)


def test_inverted_index_basic_tf_and_df():
    idx = InvertedIndex(name="all")
    idx.add_document("d1", ["python", "guide", "python"])
    idx.add_document("d2", ["python", "tutorial"])
    idx.add_document("d3", ["bash", "guide"])

    assert idx.n_docs == 3
    assert idx.term_to_postings["python"] == {"d1": 2, "d2": 1}
    assert idx.df["python"] == 2
    assert idx.df["guide"] == 2
    assert idx.df["bash"] == 1

    assert idx.doc_lens == {"d1": 3, "d2": 2, "d3": 2}
    assert idx.total_tokens == 7
    assert abs(idx.avgdl - 7 / 3) < 1e-9
    assert idx.vocab_size == 4


def test_inverted_index_handles_empty_document():
    idx = InvertedIndex(name="all")
    idx.add_document("d1", ["python"])
    idx.add_document("d2", [])  # empty doc must still increment n_docs

    assert idx.n_docs == 2
    assert idx.doc_lens["d2"] == 0
    assert "d2" not in idx.term_to_postings.get("python", {})


def test_postings_lookup_for_missing_term():
    idx = InvertedIndex(name="all")
    idx.add_document("d1", ["python"])
    assert idx.postings("nonexistent") == {}


def test_doc_store_roundtrip():
    store = DocStore()
    store.add(
        DocMetadata(
            doc_id="d1",
            post_id="p1",
            subreddit="science",
            title="t",
            selftext_excerpt="x",
            url="u",
            permalink="/p",
            score=10,
            title_len=1,
            body_len=2,
            comments_len=3,
            doc_len=6,
        )
    )
    assert len(store) == 1
    got = store.get("d1")
    assert got is not None and got.doc_id == "d1"
    assert store.get("missing") is None


def _write_processed(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_processed_record(doc_id: str, subreddit: str, title_tokens, body_tokens, comments_tokens):
    return {
        "doc_id": doc_id,
        "post_id": doc_id.split("_")[-1],
        "subreddit": subreddit,
        "title": "title text",
        "selftext_excerpt": "body text",
        "url": "https://reddit.com/x",
        "permalink": f"/r/{subreddit}/{doc_id}/",
        "score": 10,
        "title_tokens": title_tokens,
        "body_tokens": body_tokens,
        "comments_tokens": comments_tokens,
        "all_tokens": title_tokens + body_tokens + comments_tokens,
        "title_len": len(title_tokens),
        "body_len": len(body_tokens),
        "comments_len": len(comments_tokens),
        "doc_len": len(title_tokens) + len(body_tokens) + len(comments_tokens),
        "preprocessing": {"lowercase": True, "remove_stopwords": True, "stem": False, "stemmer": "none"},
    }


def test_builder_end_to_end_per_field_indexes(tmp_path: Path):
    proc_dir = tmp_path / "processed"
    _write_processed(
        proc_dir / "science.jsonl",
        [
            _make_processed_record("science_a1", "science", ["climate"], ["rising", "sea"], ["data"]),
            _make_processed_record("science_a2", "science", ["climate"], ["temperature"], []),
        ],
    )
    _write_processed(
        proc_dir / "programming.jsonl",
        [
            _make_processed_record("programming_b1", "programming", ["python"], ["async"], ["bug"]),
        ],
    )

    cfg = IndexConfig(input_dir=proc_dir, output_dir=tmp_path / "index")
    artifacts = build_artifacts(cfg)

    # All four indexes present and counted correctly
    assert set(artifacts.indexes.keys()) == {"all", "title", "body", "comments"}
    assert artifacts.all_index.n_docs == 3
    assert artifacts.indexes["title"].df["climate"] == 2
    assert artifacts.indexes["body"].df["async"] == 1
    assert artifacts.indexes["comments"].df["bug"] == 1
    assert "climate" not in artifacts.indexes["body"].df  # not a body term
    assert artifacts.all_index.df["climate"] == 2

    # DocStore populated with metadata
    assert len(artifacts.doc_store) == 3
    meta = artifacts.doc_store.get("science_a1")
    assert meta is not None and meta.subreddit == "science"


def test_save_load_roundtrip_matches_stats(tmp_path: Path):
    proc_dir = tmp_path / "processed"
    _write_processed(
        proc_dir / "test.jsonl",
        [
            _make_processed_record("test_a", "test", ["foo"], ["bar"], ["baz"]),
            _make_processed_record("test_b", "test", ["foo", "qux"], [], []),
        ],
    )
    out_dir = tmp_path / "index"

    cfg = IndexConfig(input_dir=proc_dir, output_dir=out_dir)
    artifacts = build_artifacts(cfg)
    save_artifacts(artifacts, out_dir)

    reloaded = load_artifacts(out_dir)
    assert reloaded.stats() == artifacts.stats()
    assert reloaded.indexes["all"].df["foo"] == 2
    assert reloaded.doc_store.get("test_b").title_len == 2

    manifest = load_manifest(out_dir)
    assert manifest["n_docs"] == 2
    assert manifest["indexes"]["all"]["n_docs"] == 2
    assert "config" in manifest


def test_index_config_rejects_missing_all_field():
    import pytest

    cfg = IndexConfig(fields=["title", "body"])
    with pytest.raises(ValueError):
        cfg.validate()
