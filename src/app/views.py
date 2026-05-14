"""HTTP routes for the search UI."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, List

from flask import Blueprint, Response, abort, current_app, render_template, request

from .highlight import make_snippet


main = Blueprint("main", __name__)

_CSV_COLUMNS_SEARCH = [
    "rank",
    "ir_score",
    "doc_id",
    "post_id",
    "subreddit",
    "title",
    "reddit_score",
    "permalink",
    "url",
    "snippet",
]

_CSV_COLUMNS_COMPARE = ["ranker"] + _CSV_COLUMNS_SEARCH


def _filename_stem(query: str, label: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    base = "".join("_" if c in bad else c for c in query[:80]).strip("._ ")
    base = re.sub(r"_+", "_", base)
    if not base:
        base = "export"
    safe_label = re.sub(r"[^\w.-]+", "", label)[:24] or "out"
    stem = f"reddit_ir_{safe_label}_{base}"
    return stem[:140]


def _row_from_enriched_item(item: dict) -> Dict[str, Any]:
    m = item["meta"]
    return {
        "rank": item["rank"],
        "ir_score": item["score"],
        "doc_id": m.doc_id,
        "post_id": m.post_id,
        "subreddit": m.subreddit,
        "title": m.title,
        "reddit_score": m.score,
        "permalink": m.permalink,
        "url": m.url,
        "snippet": item.get("snippet") or "",
    }


def _csv_response(rows: List[Dict[str, Any]], fieldnames: List[str], stem: str) -> Response:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    data = ("\ufeff" + buf.getvalue()).encode("utf-8")
    return Response(
        data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'},
    )


def _json_response(payload: dict, stem: str) -> Response:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        data,
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}.json"'},
    )

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


@main.route("/search/export")
def search_export():
    fmt = (request.args.get("format") or "csv").lower()
    if fmt not in ("csv", "json"):
        abort(400, "format must be csv or json")

    cfg = current_app.config["APP_CONFIG"]
    pool = current_app.config["RANKER_POOL"]

    query = (request.args.get("q") or "").strip()
    ranker_name = request.args.get("ranker") or cfg.default_ranker
    top_k = _clamp_top_k(request.args.get("k"))

    if not query:
        abort(400, "Missing query parameter q")

    if ranker_name not in pool.available:
        abort(400, f"Unknown ranker: {ranker_name!r}")

    ranker = pool.get(ranker_name)
    results = ranker.search(query, top_k=top_k)
    enriched = _enrich_results(results, query, with_snippet=True)
    rows = [_row_from_enriched_item(x) for x in enriched]

    stem = _filename_stem(query, f"search_{ranker_name}")

    if fmt == "json":
        payload = {
            "query": query,
            "ranker": ranker_name,
            "top_k": top_k,
            "n_results": len(rows),
            "results": rows,
        }
        return _json_response(payload, stem)

    return _csv_response(rows, _CSV_COLUMNS_SEARCH, stem)


@main.route("/compare/export")
def compare_export():
    fmt = (request.args.get("format") or "csv").lower()
    if fmt not in ("csv", "json"):
        abort(400, "format must be csv or json")

    cfg = current_app.config["APP_CONFIG"]
    pool = current_app.config["RANKER_POOL"]

    query = (request.args.get("q") or "").strip()
    top_k = _clamp_top_k(request.args.get("k"))

    if not query:
        abort(400, "Missing query parameter q")

    flat_rows: List[Dict[str, Any]] = []
    by_ranker: Dict[str, List[Dict[str, Any]]] = {}

    for name in pool.available:
        ranker = pool.get(name)
        results = ranker.search(query, top_k=top_k)
        enriched = _enrich_results(results, query, with_snippet=True)
        part: List[Dict[str, Any]] = []
        for item in enriched:
            row = _row_from_enriched_item(item)
            row["ranker"] = name
            flat_rows.append(row)
            part.append({k: v for k, v in row.items() if k != "ranker"})
        by_ranker[name] = part

    stem = _filename_stem(query, "compare")

    if fmt == "json":
        payload = {
            "query": query,
            "top_k": top_k,
            "rankers": list(pool.available),
            "n_rows": len(flat_rows),
            "results": flat_rows,
            "by_ranker": by_ranker,
        }
        return _json_response(payload, stem)

    return _csv_response(flat_rows, _CSV_COLUMNS_COMPARE, stem)
