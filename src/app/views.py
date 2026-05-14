"""HTTP routes for the search UI."""

from __future__ import annotations

import html as html_stdlib
import re
from dataclasses import asdict
from typing import Any, Dict, List

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from .highlight import make_snippet


_SNIP_TAG_RE = re.compile(r"<[^>]+>")


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


def _snippet_plain_from_markup(snippet: object) -> str:
    """Strip `<mark>` / entities from snippet Markup for JSON export."""
    if snippet is None:
        return ""
    s = str(snippet)
    s = _SNIP_TAG_RE.sub("", s)
    return html_stdlib.unescape(s)


def _json_rows_from_enriched(enriched: List[dict]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in enriched:
        meta = item["meta"]
        row: Dict[str, Any] = {
            "rank": item["rank"],
            "score": item["score"],
            "meta": asdict(meta),
        }
        if "snippet" in item:
            row["snippet"] = _snippet_plain_from_markup(item["snippet"])
        rows.append(row)
    return rows


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


@main.route("/search/export")
def search_export():
    """Return the current search results as JSON (optional `download=1` attachment)."""
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
    rows = _json_rows_from_enriched(enriched)
    payload = {
        "query": query,
        "ranker": ranker_name,
        "top_k": top_k,
        "n_results": len(rows),
        "results": rows,
    }
    resp = jsonify(payload)
    if request.args.get("download"):
        resp.headers["Content-Disposition"] = 'attachment; filename="search-results.json"'
    return resp


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


@main.route("/compare/export")
def compare_export():
    """Return side-by-side compare results for all rankers as JSON."""
    cfg = current_app.config["APP_CONFIG"]
    pool = current_app.config["RANKER_POOL"]

    query = (request.args.get("q") or "").strip()
    top_k = _clamp_top_k(request.args.get("k"))

    rankers_payload: Dict[str, Any] = {}
    if query:
        for name in pool.available:
            ranker = pool.get(name)
            results = ranker.search(query, top_k=top_k)
            enriched = _enrich_results(results, query, with_snippet=True)
            rankers_payload[name] = _json_rows_from_enriched(enriched)

    payload = {
        "query": query,
        "top_k": top_k,
        "default_ranker": cfg.default_ranker,
        "n_rankers": len(rankers_payload),
        "rankers": rankers_payload,
    }
    resp = jsonify(payload)
    if request.args.get("download"):
        resp.headers["Content-Disposition"] = 'attachment; filename="compare-results.json"'
    return resp
