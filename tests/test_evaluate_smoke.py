"""End-to-end smoke tests for the evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluate.pool import generate_pool
from src.evaluate.qrels import load_qrels_csv, save_pool_csv
from src.evaluate.queries import Query, load_queries, save_queries
from src.evaluate.runner import run_evaluation, write_results_csvs
from src.index.builder import build_artifacts
from src.index.config import IndexConfig
from src.rankers.bm25_ranker import Bm25Ranker
from src.rankers.tfidf_ranker import TfidfRanker


def _write_processed(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(doc_id, subreddit, title_tokens, body_tokens, comments_tokens, *, title="t", excerpt="e"):
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
        "selftext_excerpt": excerpt,
        "url": "https://reddit.com/x",
        "permalink": f"/r/{subreddit}/{doc_id}/",
        "score": 1,
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


def _build_artifacts(tmp_path: Path):
    proc_dir = tmp_path / "processed"
    _write_processed(
        proc_dir / "test.jsonl",
        [
            _record("d1", "test", ["python"], ["python", "guide"], [], title="Python guide"),
            _record("d2", "test", ["python"], ["async", "websocket"], [], title="Async websocket"),
            _record("d3", "test", ["bash"], ["guide"], [], title="Bash guide"),
            _record("d4", "test", ["climate"], ["sea", "rising"], [], title="Climate study"),
        ],
    )
    cfg = IndexConfig(input_dir=proc_dir, output_dir=tmp_path / "index")
    return build_artifacts(cfg)


def test_query_set_roundtrip(tmp_path: Path):
    qpath = tmp_path / "queries.json"
    save_queries(
        [Query(qid="q1", text="python", category="technical"), Query(qid="q2", text="climate")],
        qpath,
    )
    loaded = load_queries(qpath)
    assert [q.qid for q in loaded] == ["q1", "q2"]
    assert loaded[0].category == "technical"


def test_pool_generation_is_deterministic_and_complete(tmp_path: Path):
    artifacts = _build_artifacts(tmp_path)
    queries = [Query(qid="q1", text="python"), Query(qid="q2", text="climate")]
    rankers = [
        ("tfidf_cosine", TfidfRanker(artifacts)),
        ("bm25_plain", Bm25Ranker(artifacts)),
    ]

    rows1 = generate_pool(artifacts, rankers, queries, pool_depth=10, shuffle_seed=7)
    rows2 = generate_pool(artifacts, rankers, queries, pool_depth=10, shuffle_seed=7)
    # Same seed → same shuffled order.
    assert [r["doc_id"] for r in rows1] == [r["doc_id"] for r in rows2]

    # Pool covers both queries.
    qids = {r["qid"] for r in rows1}
    assert qids == {"q1", "q2"}

    # Schema columns present and labels empty.
    expected_cols = {"qid", "query", "doc_id", "subreddit", "title", "excerpt", "label"}
    assert expected_cols <= set(rows1[0].keys())
    assert all(r["label"] == "" for r in rows1)


def test_qrels_csv_roundtrip(tmp_path: Path):
    rows = [
        {"qid": "q1", "query": "python", "doc_id": "d1", "subreddit": "test",
         "title": "t", "excerpt": "e", "label": 1},
        {"qid": "q1", "query": "python", "doc_id": "d2", "subreddit": "test",
         "title": "t", "excerpt": "e", "label": 0},
        {"qid": "q1", "query": "python", "doc_id": "d3", "subreddit": "test",
         "title": "t", "excerpt": "e", "label": ""},  # blank → 0
    ]
    path = tmp_path / "qrels.csv"
    save_pool_csv(rows, path)
    qrels = load_qrels_csv(path)
    assert qrels == {"q1": {"d1": 1, "d2": 0, "d3": 0}}


def test_run_evaluation_end_to_end(tmp_path: Path):
    artifacts = _build_artifacts(tmp_path)
    queries = [Query(qid="q1", text="python"), Query(qid="q2", text="climate")]

    # Hand-built qrels: d1 and d2 relevant for "python", d4 relevant for "climate".
    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d4": 1}}

    rankers = [("bm25_plain", Bm25Ranker(artifacts))]
    results = run_evaluation(rankers, queries, qrels, top_k=5)

    bm25 = results["per_ranker"]["bm25_plain"]
    assert bm25["aggregate"]["n_queries"] == 2
    # BM25 should retrieve the relevant docs in top-5 → P@5 should be > 0.
    assert bm25["aggregate"]["P@5"] > 0
    assert bm25["aggregate"]["MAP"] > 0

    # write_results_csvs produces both CSVs.
    out = tmp_path / "results"
    per_query, summary = write_results_csvs(results, out)
    assert per_query.exists()
    assert summary.exists()
    summary_text = summary.read_text(encoding="utf-8")
    assert "ranker" in summary_text and "MAP" in summary_text


def test_evaluate_build_rankers_dedupes_legacy_aliases(tmp_path: Path):
    from src.evaluate.cli import _build_rankers_from_names

    artifacts = _build_artifacts(tmp_path)
    rankers = _build_rankers_from_names(artifacts, ["tfidf", "tfidf_cosine", "bm25", "bm25_plain"])
    assert [n for n, _ in rankers] == ["tfidf_cosine", "bm25_plain"]
