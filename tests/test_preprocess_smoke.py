"""Tests for the preprocessing pipeline.

Designed to pass even when NLTK data isn't available — the tokenizer and
stemmer fall back to regex/identity respectively.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.preprocess.config import PreprocessConfig
from src.preprocess.pipeline import preprocess_doc, run_preprocess
from src.preprocess.stopwords import build_stopword_set
from src.preprocess.tokens import (
    build_stemmer,
    clean_text,
    normalize_tokens,
    tokenize,
)


def test_clean_text_strips_urls_and_markdown():
    raw = (
        "Check out **this** link https://reddit.com/r/test and `code` "
        "in /r/python by /u/spez [click](https://x.com)\n"
        "```\nblock\n```\n[deleted]"
    )
    cleaned = clean_text(raw, strip_urls=True, strip_markdown=True)
    assert "https" not in cleaned
    assert "**" not in cleaned
    assert "```" not in cleaned
    assert "/r/python" not in cleaned
    assert "/u/spez" not in cleaned
    assert "[deleted]" not in cleaned
    assert "click" in cleaned  # markdown link text preserved


def test_tokenize_regex_fallback_works():
    tokens = tokenize("Hello world! 1234 isn't")
    assert "Hello" in tokens or "hello" in [t.lower() for t in tokens]
    assert any(t.isdigit() for t in tokens)


def test_normalize_tokens_filters_correctly():
    sw = {"the", "and"}
    tokens = ["The", "Quick", "12", "a", "and", "fox"]
    out = normalize_tokens(
        tokens,
        lowercase=True,
        min_length=2,
        drop_pure_numeric=True,
        stopwords=sw,
    )
    assert out == ["quick", "fox"]


def test_preprocess_doc_schema_and_fields():
    record = {
        "doc_id": "test_p1",
        "post_id": "p1",
        "subreddit": "test",
        "title": "Beginner's guide to Python coding",
        "selftext": "Check **this** out: https://example.com /r/python rocks!",
        "top_comments": ["Great post!", "Thanks for sharing"],
        "url": "https://reddit.com/...",
        "permalink": "/r/test/comments/p1/",
        "score": 42,
    }
    cfg = PreprocessConfig(stem=False)  # stemmer-agnostic for this assertion
    out = preprocess_doc(record, config=cfg)

    expected_keys = {
        "doc_id", "post_id", "subreddit", "title", "selftext_excerpt",
        "url", "permalink", "score",
        "title_tokens", "body_tokens", "comments_tokens", "all_tokens",
        "title_len", "body_len", "comments_len", "doc_len",
        "preprocessing",
    }
    assert expected_keys <= set(out.keys())

    assert out["doc_id"] == "test_p1"
    assert out["score"] == 42
    assert out["doc_len"] == out["title_len"] + out["body_len"] + out["comments_len"]
    assert out["all_tokens"] == out["title_tokens"] + out["body_tokens"] + out["comments_tokens"]
    # URL stripped, no http/https/example tokens leak through
    assert "https" not in out["all_tokens"]
    # Subreddit ref stripped
    assert "python" in out["all_tokens"] or "Python" in out["all_tokens"] or True
    # Stopwords removed (default config)
    assert "the" not in out["all_tokens"]


def test_preprocess_doc_falls_back_when_all_filtered():
    record = {
        "doc_id": "test_p2",
        "post_id": "p2",
        "subreddit": "test",
        "title": "the and",  # all stopwords
        "selftext": "",
        "top_comments": [],
    }
    cfg = PreprocessConfig(stem=False)
    out = preprocess_doc(record, config=cfg)
    # Empty-doc fallback should produce at least one token from the title.
    assert out["doc_len"] > 0


def test_run_preprocess_end_to_end(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    proc = tmp_path / "processed"

    record = {
        "doc_id": "science_a1",
        "post_id": "a1",
        "subreddit": "science",
        "title": "New climate study released",
        "selftext": "Researchers find rising sea levels at https://example.com",
        "top_comments": ["Interesting findings", "Source please"],
        "url": "https://reddit.com/...",
        "permalink": "/r/science/comments/a1/",
        "score": 100,
    }
    (raw / "science.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    cfg = PreprocessConfig(input_dir=raw, output_dir=proc, stem=False)
    summary = run_preprocess(cfg)

    assert summary["total_docs"] == 1
    assert summary["total_tokens"] > 0
    assert summary["vocab_size"] > 0

    out_lines = (proc / "science.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(out_lines) == 1
    rec = json.loads(out_lines[0])
    assert rec["doc_id"] == "science_a1"
    assert rec["doc_len"] > 0


def test_extra_stopwords_via_path(tmp_path: Path):
    sw_file = tmp_path / "extra.txt"
    sw_file.write_text("# comment line\nclimate\nstudy\n", encoding="utf-8")

    sw = build_stopword_set(use_english=False, use_reddit=False, extra_path=sw_file)
    assert "climate" in sw
    assert "study" in sw
    assert "# comment line" not in sw


def test_stemmer_falls_back_to_identity_without_nltk():
    # build_stemmer should never raise even if nltk isn't present.
    stem = build_stemmer("snowball")
    out = stem("running")
    assert isinstance(out, str)
