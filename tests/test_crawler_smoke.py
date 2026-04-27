"""End-to-end smoke tests for the crawler that don't touch the Reddit API.

These tests use lightweight fakes that mimic ``praw.Reddit``,
``praw.models.Subreddit``, ``praw.models.Submission`` and
``praw.models.CommentForest`` just enough to exercise the runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.crawler.config import CrawlerConfig
from src.crawler.runner import Runner


class FakeComment:
    def __init__(self, body: str):
        self.body = body


class FakeCommentForest:
    def __init__(self, comments):
        self._comments = list(comments)

    def __iter__(self):
        return iter(self._comments)

    def replace_more(self, limit: int = 0) -> None:
        return None


class FakeSubmission:
    def __init__(
        self,
        id: str,
        title: str,
        selftext: str = "",
        score: int = 10,
        comments=None,
        over_18: bool = False,
    ):
        self.id = id
        self.title = title
        self.selftext = selftext
        self.score = score
        self.url = f"https://reddit.com/{id}"
        self.permalink = f"/r/test/comments/{id}/"
        self.created_utc = 1_700_000_000.0
        self.num_comments = 5
        self.over_18 = over_18
        self.is_self = True
        self.comments = FakeCommentForest(comments or [])


class FakeSubredditPage:
    def __init__(self, submissions):
        self._submissions = submissions

    def _slice(self, limit):
        return iter(self._submissions[:limit] if limit else self._submissions)

    def top(self, *, time_filter="all", limit=None):
        return self._slice(limit)

    def hot(self, *, limit=None):
        return self._slice(limit)

    def new(self, *, limit=None):
        return self._slice(limit)

    def rising(self, *, limit=None):
        return self._slice(limit)


class FakeReddit:
    def __init__(self, subreddits_data):
        self._data = subreddits_data

    def subreddit(self, name):
        return FakeSubredditPage(self._data.get(name, []))


def _build_config(tmp_path: Path, **overrides) -> CrawlerConfig:
    base = dict(
        subreddits=["test"],
        output_dir=tmp_path / "raw",
        sort_mode="top",
        time_filter="month",
        post_limit=10,
        comment_limit=2,
        batch_sleep_ms=0,
        jitter_ms=0,
        max_retries=1,
        checkpoint_every=2,
        client_id="x",
        client_secret="y",
        user_agent="test",
    )
    base.update(overrides)
    return CrawlerConfig(**base)


def _factory(reddit):
    return lambda _cfg: reddit


def _noop_sleep(_seconds: float) -> None:
    return None


def test_writes_records_and_manifest(tmp_path):
    submissions = [
        FakeSubmission(
            "p1",
            "Hello",
            "world",
            comments=[FakeComment("c1"), FakeComment("c2"), FakeComment("c3")],
        ),
        FakeSubmission("p2", "Foo", "bar"),
    ]
    fake = FakeReddit({"test": submissions})

    cfg = _build_config(tmp_path)
    stats = Runner(cfg, reddit_factory=_factory(fake), sleeper=_noop_sleep).run()

    assert stats["saved"] == 2
    assert stats["skipped_dedup"] == 0
    assert stats["skipped_filter"] == 0

    jsonl_path = tmp_path / "raw" / "test.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    rec = json.loads(lines[0])
    assert rec["doc_id"] == "test_p1"
    assert rec["subreddit"] == "test"
    assert rec["title"] == "Hello"
    assert rec["top_comments"] == ["c1", "c2"]  # capped at comment_limit=2

    manifest = json.loads((tmp_path / "raw" / "crawl_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 1
    assert manifest["runs"][0]["stats"]["saved"] == 2
    assert "client_id" not in manifest["runs"][0]["params"]


def test_resume_is_idempotent(tmp_path):
    submissions = [
        FakeSubmission("p1", "Hello", "world"),
        FakeSubmission("p2", "Foo", "bar"),
    ]

    cfg = _build_config(tmp_path)
    Runner(
        cfg,
        reddit_factory=_factory(FakeReddit({"test": submissions})),
        sleeper=_noop_sleep,
    ).run()

    stats2 = Runner(
        cfg,
        reddit_factory=_factory(FakeReddit({"test": submissions})),
        sleeper=_noop_sleep,
    ).run()

    assert stats2["saved"] == 0
    assert stats2["skipped_dedup"] == 2

    lines = (tmp_path / "raw" / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_filters_skip_removed_low_score_and_nsfw(tmp_path):
    submissions = [
        FakeSubmission("good", "Real title", "body", score=100),
        FakeSubmission("removed", "", "", score=200),
        FakeSubmission("lowscore", "ok", "ok", score=1),
        FakeSubmission("nsfw", "spicy", "body", score=200, over_18=True),
    ]
    cfg = _build_config(tmp_path, min_score=10)
    stats = Runner(
        cfg,
        reddit_factory=_factory(FakeReddit({"test": submissions})),
        sleeper=_noop_sleep,
    ).run()

    assert stats["saved"] == 1
    assert stats["skipped_filter"] == 3


def test_retry_then_success(tmp_path):
    """A submission whose comment fetch fails once should still be saved."""

    class FlakyForest(FakeCommentForest):
        def __init__(self, comments):
            super().__init__(comments)
            self._calls = 0

        def replace_more(self, limit: int = 0) -> None:
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("transient network blip")
            return None

    flaky = FakeSubmission("p1", "title", "body")
    flaky.comments = FlakyForest([FakeComment("ok")])

    cfg = _build_config(tmp_path, max_retries=2)
    stats = Runner(
        cfg,
        reddit_factory=_factory(FakeReddit({"test": [flaky]})),
        sleeper=_noop_sleep,
    ).run()

    # replace_more failure is swallowed inside extract_top_comments,
    # so the post is saved on the first try.
    assert stats["saved"] == 1
