"""Tests for the Flask UI using Flask's test client.

Each test builds a small index in tmp_path and constructs a fresh Flask app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")  # gracefully skip if Flask isn't installed

from src.app import AppConfig, create_app  # noqa: E402
from src.index.builder import build_artifacts  # noqa: E402
from src.index.config import IndexConfig  # noqa: E402
from src.index.store import save_artifacts  # noqa: E402


def _record(doc_id, subreddit, title, title_tokens, body_tokens, comments_tokens, *, score=10):
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
        "post_id": doc_id,
        "subreddit": subreddit,
        "title": title,
        "selftext_excerpt": " ".join(body_tokens),
        "url": f"https://reddit.com/{doc_id}",
        "permalink": f"/r/{subreddit}/{doc_id}/",
        "score": score,
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
def app(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    records = [
        _record("d1", "test", "Beginner Python guide", ["python"], ["python", "guide"], []),
        _record("d2", "test", "Async websocket bug", ["async", "websocket"], ["python"], ["bug"]),
        _record("d3", "test", "Climate study", ["climate"], ["sea", "rising"], []),
    ]
    with open(proc / "test.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = IndexConfig(input_dir=proc, output_dir=tmp_path / "index")
    artifacts = build_artifacts(cfg)
    save_artifacts(artifacts, tmp_path / "index")

    app_cfg = AppConfig(index_dir=tmp_path / "index", default_ranker="bm25_plain")
    app = create_app(app_cfg)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Reddit IR" in r.data
    assert b"Indexed" in r.data


def test_search_returns_expected_doc(client):
    r = client.get("/search?q=python")
    assert r.status_code == 200
    body = r.data.decode("utf-8")
    assert "Beginner Python guide" in body
    # Snippet should highlight the matched stem.
    assert "<mark>" in body


def test_search_empty_query_renders_gracefully(client):
    r = client.get("/search?q=")
    assert r.status_code == 200
    assert b"Enter a query" in r.data


def test_search_oov_query_shows_no_results(client):
    r = client.get("/search?q=zzznotaword")
    assert r.status_code == 200
    assert b"No results" in r.data


def test_compare_page_lists_all_default_rankers(client):
    from src.rankers.registry import DEFAULT_RANKER_NAMES

    r = client.get("/compare?q=python")
    assert r.status_code == 200
    body = r.data.decode("utf-8")
    for ranker_name in DEFAULT_RANKER_NAMES:
        assert ranker_name in body


def test_unknown_ranker_returns_400(client):
    r = client.get("/search?q=python&ranker=mystery")
    assert r.status_code == 400


def test_top_k_clamps_to_max(client, app):
    huge = app.config["APP_CONFIG"].max_top_k + 100
    r = client.get(f"/search?q=python&k={huge}")
    assert r.status_code == 200  # no crash, just clamps
