"""HTTP routes for the search UI."""

from __future__ import annotations

from typing import List

from flask import Blueprint, abort, current_app, render_template, request

from .highlight import make_snippet


main = Blueprint("main", __name__)

# UI + URL only allow these top-k values (must be ≤ AppConfig.max_top_k).
_ALLOWED_TOP_K = (10, 20, 30)


def _clamp_top_k(raw: str | None) -> int:
    cfg = current_app.config["APP_CONFIG"]
    default = cfg.default_top_k if cfg.default_top_k in _ALLOWED_TOP_K else _ALLOWED_TOP_K[0]
    if raw is None:
        return default
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return default
    if k in _ALLOWED_TOP_K:
        return k
    return default


def _enrich_results(results, query: str, *, with_snippet: bool):
    artifacts = current_app.config["INDEX_ARTIFACTS"]
    preprocessing = artifacts.preprocessing or {}
    enriched = []
    for r in results:
        meta = artifacts.doc_store.get(r.doc_id)
        if meta is None:
            continue
        item = {"rank": r.rank, "score": r.score, "meta": meta}
        if with_snippet:
            source = meta.selftext_excerpt or meta.title
            item["snippet"] = make_snippet(source, query, preprocessing)
        enriched.append(item)
    return enriched


@main.route("/")
def index():
    cfg = current_app.config["APP_CONFIG"]
    return render_template(
        "index.html",
        query="",
        current_ranker=cfg.default_ranker,
        current_top_k=cfg.default_top_k,
    )


@main.route("/search")
def search():
    cfg = current_app.config["APP_CONFIG"]
    pool = current_app.config["RANKER_POOL"]

    query = (request.args.get("q") or "").strip()
    ranker_name = request.args.get("ranker") or cfg.default_ranker
    top_k = _clamp_top_k(request.args.get("k"))

    if ranker_name not in pool.available:
        abort(400, f"Unknown ranker: {ranker_name!r}")

    results: List = []
    if query:
        ranker = pool.get(ranker_name)
        results = ranker.search(query, top_k=top_k)

    enriched = _enrich_results(results, query, with_snippet=True)
    return render_template(
        "results.html",
        query=query,
        current_ranker=ranker_name,
        top_k=top_k,
        current_top_k=top_k,
        results=enriched,
    )


@main.route("/compare")
def compare():
    cfg = current_app.config["APP_CONFIG"]
    pool = current_app.config["RANKER_POOL"]

    query = (request.args.get("q") or "").strip()
    top_k = _clamp_top_k(request.args.get("k"))

    columns = []
    if query:
        for name in pool.available:
            ranker = pool.get(name)
            results = ranker.search(query, top_k=top_k)
            columns.append(
                {
                    "name": name,
                    "results": _enrich_results(results, query, with_snippet=False),
                }
            )

    return render_template(
        "compare.html",
        query=query,
        current_ranker=cfg.default_ranker,
        top_k=top_k,
        current_top_k=top_k,
        columns=columns,
    )
