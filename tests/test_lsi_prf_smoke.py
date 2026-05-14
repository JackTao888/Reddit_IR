"""Smoke tests for LSI (SVD) and BM25+PRF rankers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.index.builder import build_artifacts
from src.index.config import IndexConfig
from src.rankers.registry import build_ranker


def _write_processed(path: Path, records):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(doc_id, subreddit, title_tokens, body_tokens, comments_tokens, *, title="t"):
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
    excerpt = " ".join(body_tokens)
    return {
        "doc_id": doc_id,
        "post_id": doc_id,
        "subreddit": subreddit,
        "title": title,
        "selftext_excerpt": excerpt,
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
def artifacts_6docs(tmp_path: Path):
    proc_dir = tmp_path / "processed"
    _write_processed(
        proc_dir / "test.jsonl",
        [
            _record("d1", "test", ["python"], ["python", "guide"], [], title="Python guide"),
            _record("d2", "test", ["python"], ["tutorial", "async"], [], title="Python async"),
            _record("d3", "test", ["bash"], ["guide", "shell"], [], title="Bash guide"),
            _record("d4", "test", ["climate"], ["sea", "rising"], [], title="Climate"),
            _record("d5", "test", ["data"], ["python", "pandas"], [], title="Data pandas"),
            _record("d6", "test", ["web"], ["http", "rest"], [], title="Web rest"),
        ],
    )
    cfg = IndexConfig(input_dir=proc_dir, output_dir=tmp_path / "index")
    return build_artifacts(cfg)


def test_bm25_prf_returns_results(artifacts_6docs):
    r = build_ranker("bm25_prf", artifacts_6docs)
    assert r.name == "bm25_prf"
    out = r.search("python", top_k=5)
    assert out
    assert len(out) <= 5


def test_lsi_ranker_runs_with_scipy(artifacts_6docs):
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    r = build_ranker("lsi", artifacts_6docs, lsi_k=4, lsi_max_vocab=500)
    assert "lsi" in r.name
    out = r.search("python", top_k=5)
    assert out
