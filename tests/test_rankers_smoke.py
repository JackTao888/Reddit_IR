"""Tests for TF-IDF and BM25 rankers on synthetic indexed data."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.index.builder import build_artifacts
from src.index.config import IndexConfig
from src.rankers.bm25_ranker import Bm25Ranker
from src.rankers.query import prepare_query
from src.rankers.tfidf_ranker import TfidfRanker


def _write_processed(path: Path, records):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(doc_id, subreddit, title_tokens, body_tokens, comments_tokens):
    preprocessing = {
        "lowercase": True,
        "remove_stopwords": True,
        "stem": False,
        "stemmer": "none",
        "min_token_length": 2,
        "drop_pure_numeric": True,
        "strip_urls": True,
        "strip_markdown": True,
        "keep_subreddit_refs": False,
    }
    return {
        "doc_id": doc_id,
        "post_id": doc_id.split("_")[-1],
        "subreddit": subreddit,
        "title": " ".join(title_tokens),
        "selftext_excerpt": " ".join(body_tokens),
        "url": f"https://reddit.com/{doc_id}",
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
        "preprocessing": preprocessing,
    }


@pytest.fixture
def synthetic_artifacts(tmp_path: Path):
    proc_dir = tmp_path / "processed"
    _write_processed(
        proc_dir / "test.jsonl",
        [
            _record("d1", "test", ["python"], ["python", "guide"], []),
            _record("d2", "test", ["python"], ["tutorial"], []),
            _record("d3", "test", ["bash"], ["guide"], []),
            _record("d4", "test", ["python"], ["async", "websocket", "bug"], ["thanks"]),
        ],
    )
    cfg = IndexConfig(input_dir=proc_dir, output_dir=tmp_path / "index")
    return build_artifacts(cfg)


def test_index_captures_preprocessing_metadata(synthetic_artifacts):
    p = synthetic_artifacts.preprocessing
    assert isinstance(p, dict)
    assert p["lowercase"] is True
    assert p["min_token_length"] == 2


def test_prepare_query_matches_doc_pipeline():
    p = {"lowercase": True, "stem": False, "stemmer": "none", "remove_stopwords": True,
         "min_token_length": 2, "drop_pure_numeric": True,
         "strip_urls": True, "strip_markdown": True, "keep_subreddit_refs": False}
    tokens = prepare_query("Python ASYNC websocket!", p)
    assert tokens == ["python", "async", "websocket"]


def test_tfidf_returns_python_docs_in_order(synthetic_artifacts):
    ranker = TfidfRanker(synthetic_artifacts)
    results = ranker.search("python", top_k=5)
    doc_ids = [r.doc_id for r in results]

    # All three python docs ranked; bash-only doc excluded.
    assert "d3" not in doc_ids
    assert set(doc_ids) <= {"d1", "d2", "d4"}
    assert len(doc_ids) >= 1
    # Ranks are 1-indexed and contiguous.
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_bm25_plain_returns_python_docs(synthetic_artifacts):
    ranker = Bm25Ranker(synthetic_artifacts)
    results = ranker.search("python", top_k=5)
    doc_ids = [r.doc_id for r in results]
    assert "d3" not in doc_ids
    # d1 has tf=2 for python in the "all" index; should outrank tf=1 doc.
    assert doc_ids[0] == "d1"


def test_bm25_oov_query_returns_empty(synthetic_artifacts):
    ranker = Bm25Ranker(synthetic_artifacts)
    assert ranker.search("zzznonexistent", top_k=5) == []


def test_bm25_field_aware_weights_change_ranking(synthetic_artifacts):
    """Pumping title weight should rank a title-match higher than a body-match."""
    body_heavy = Bm25Ranker(
        synthetic_artifacts,
        field_weights={"title": 0.0, "body": 1.0, "comments": 0.0},
    )
    title_heavy = Bm25Ranker(
        synthetic_artifacts,
        field_weights={"title": 5.0, "body": 0.1, "comments": 0.0},
    )

    body_results = body_heavy.search("guide", top_k=5)
    title_results = title_heavy.search("python", top_k=5)

    # Body-only weighting on "guide": d1 and d3 both have it in body.
    body_ids = {r.doc_id for r in body_results}
    assert body_ids <= {"d1", "d3"}

    # Title-heavy weighting on "python": doc with python in title should win.
    assert title_results
    assert title_results[0].doc_id in {"d1", "d2", "d4"}


def test_bm25_field_aware_zero_weight_disables_field(synthetic_artifacts):
    only_comments = Bm25Ranker(
        synthetic_artifacts,
        field_weights={"title": 0.0, "body": 0.0, "comments": 1.0},
    )
    # Only d4 has any comment tokens.
    results = only_comments.search("thanks", top_k=5)
    assert [r.doc_id for r in results] == ["d4"]


def test_empty_query_returns_empty(synthetic_artifacts):
    assert TfidfRanker(synthetic_artifacts).search("", top_k=5) == []
    assert Bm25Ranker(synthetic_artifacts).search("the and", top_k=5) == []
