#!/usr/bin/env python3
"""Convert Reddit CSV dumps to crawler-style JSONL for ``src.preprocess``.

Supports two CSV shapes found in this repo:

**A — Numeric / pandas export** (current ``dataset/*.csv``):

  First header row looks like: ``,0,1,2,...,10``
  - ``""``   : row index (optional; used for unique keys)
  - ``0``    : body text
  - ``1``    : Reddit short id (post/comment id without ``t1_`` prefix)
  - ``2``    : subreddit name
  - ``3``    : metareddit (ignored except you could log it)
  - ``4``    : ``created_utc`` (float seconds)
  - ``5``    : author
  - ``8``    : score (often ``ups`` / comment score in exports)

**B — Named columns** (``dataset/headers.txt`` style)::

  text,id,subreddit,meta,time,author,ups,downs,...

Each output line is one JSON object with fields compatible with
``src.preprocess.pipeline.preprocess_doc`` (title, selftext, post_id, …).

Usage (from repository root)::

    python scripts/csv_to_raw_jsonl.py --input-dir dataset --output-dir data/raw
    python scripts/csv_to_raw_jsonl.py --input dataset/gaming_minecraft.csv --output data/raw/gaming_minecraft.jsonl

Then::

    python -m src.preprocess.cli --input data/raw --output data/processed
    python -m src.index.cli build --input data/processed --output data/index
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.crawler.serializers import build_doc_id as _build_doc_id_lib
except Exception:  # pragma: no cover - allow standalone run
    _build_doc_id_lib = None


_WS_RE = re.compile(r"\s+")

# Schema A: first row of dataset CSVs
_NUMERIC_HEADER = {"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}

# Schema B: named headers (subset required)
_NAMED_REQUIRED = {"text", "id", "subreddit"}


class Schema(str, Enum):
    NUMERIC = "numeric"
    NAMED = "named"


def build_doc_id(subreddit: str, unique_post_key: str) -> str:
    """Same contract as ``src.crawler.serializers.build_doc_id``."""
    if _build_doc_id_lib is not None:
        return _build_doc_id_lib(subreddit, unique_post_key)
    return f"{(subreddit or 'unknown').strip().lower()}_{unique_post_key}"


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return _WS_RE.sub(" ", str(value)).strip()


def _int_safe(raw: str | None, default: int = 0) -> int:
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _float_safe(raw: str | None, default: float = 0.0) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def detect_schema(fieldnames: list[str] | None) -> Schema:
    if not fieldnames:
        raise ValueError("CSV has no header row.")
    keys = {f.strip() if isinstance(f, str) else str(f) for f in fieldnames}
    if _NUMERIC_HEADER.issubset(keys):
        return Schema.NUMERIC
    if _NAMED_REQUIRED.issubset(keys):
        return Schema.NAMED
    raise ValueError(
        "Unrecognized CSV columns. Expected either numeric headers "
        f"{sorted(_NUMERIC_HEADER)} or named columns including {sorted(_NAMED_REQUIRED)}. "
        f"Got: {fieldnames!r}"
    )


def row_to_record_numeric(row: dict[str, Any], *, source_stem: str) -> dict | None:
    post_id = (row.get("1") or "").strip()
    row_idx = (row.get("") or "").strip()
    sub = (row.get("2") or "unknown").strip().lower() or "unknown"
    text = _normalize_text(row.get("0"))
    author = _normalize_text(row.get("5"))

    if not post_id:
        return None
    if not text:
        return None

    tail = f"{post_id}_{row_idx}" if row_idx else post_id
    unique_key = f"{source_stem}_{tail}"
    doc_id = build_doc_id(sub, unique_key)

    title = f"{author} · r/{sub}" if author else f"r/{sub} · {post_id}"
    permalink = f"/r/{sub}/comments/{post_id}/"

    return {
        "doc_id": doc_id,
        "post_id": post_id,
        "subreddit": sub,
        "title": title[:300],
        "selftext": text,
        "top_comments": [],
        "url": f"https://www.reddit.com{permalink}",
        "permalink": permalink,
        "created_utc": _float_safe(row.get("4")),
        "score": _int_safe(row.get("8")),
        "num_comments": 0,
        "over_18": False,
        "is_self": True,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def row_to_record_named(row: dict[str, Any], *, source_stem: str, row_index: int) -> dict | None:
    post_id = (row.get("id") or "").strip()
    sub = (row.get("subreddit") or "unknown").strip().lower() or "unknown"
    text = _normalize_text(row.get("text"))
    author = _normalize_text(row.get("author"))

    if not post_id:
        return None
    if not text:
        return None

    unique_key = f"{source_stem}_{post_id}_{row_index}"
    doc_id = build_doc_id(sub, unique_key)

    title = f"{author} · r/{sub}" if author else f"r/{sub} · {post_id}"
    permalink = f"/r/{sub}/comments/{post_id}/"

    score = row.get("ups") or row.get("score") or "0"
    created = row.get("time") or row.get("created_utc") or "0"

    return {
        "doc_id": doc_id,
        "post_id": post_id,
        "subreddit": sub,
        "title": title[:300],
        "selftext": text,
        "top_comments": [],
        "url": f"https://www.reddit.com{permalink}",
        "permalink": permalink,
        "created_utc": _float_safe(created if isinstance(created, str) else str(created)),
        "score": _int_safe(score if isinstance(score, str) else str(score)),
        "num_comments": 0,
        "over_18": False,
        "is_self": True,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _open_csv(path: Path, encoding: str, errors: str):
    return open(path, newline="", encoding=encoding, errors=errors)


def convert_one_csv(
    input_path: Path,
    output_path: Path,
    *,
    source_stem: str,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> dict:
    written = 0
    skipped = 0
    schema_used: str | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_csv(input_path, encoding, errors) as inf, open(
        output_path, "w", encoding="utf-8"
    ) as outf:
        reader = csv.DictReader(inf)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path}: CSV has no header row.")

        schema = detect_schema(list(reader.fieldnames))
        schema_used = schema.value

        if schema == Schema.NUMERIC:
            for row in reader:
                rec = row_to_record_numeric(row, source_stem=source_stem)
                if rec is None:
                    skipped += 1
                    continue
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        else:
            for i, row in enumerate(reader):
                rec = row_to_record_named(row, source_stem=source_stem, row_index=i)
                if rec is None:
                    skipped += 1
                    continue
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "schema": schema_used,
        "written": written,
        "skipped": skipped,
    }


def iter_csv_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    yield from sorted(path.glob("*.csv"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=None, help="Single CSV file.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (with --input). Default: data/raw/<stem>.jsonl",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Convert every *.csv in this directory (or use --input as a directory).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Batch output directory (default: data/raw).",
    )
    p.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Input CSV encoding (default: utf-8-sig to strip BOM if present).",
    )
    p.add_argument(
        "--errors",
        default="replace",
        help="Codec error handler for input (default: replace).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print JSON summary to stdout.",
    )
    args = p.parse_args(argv)

    def emit(obj: dict) -> None:
        if not args.quiet:
            print(json.dumps(obj, indent=2, ensure_ascii=False))

    # Batch: --input-dir or --input pointing at a directory
    batch_dir = args.input_dir
    if batch_dir is None and args.input is not None and args.input.is_dir():
        batch_dir = args.input

    if batch_dir is not None:
        if not batch_dir.is_dir():
            raise SystemExit(f"Not a directory: {batch_dir}")
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_files = list(iter_csv_files(batch_dir))
        if not csv_files:
            raise SystemExit(f"No .csv files under {batch_dir}")
        summaries: list[dict] = []
        for csv_path in csv_files:
            stem = csv_path.stem
            summaries.append(
                convert_one_csv(
                    csv_path,
                    out_dir / f"{stem}.jsonl",
                    source_stem=stem,
                    encoding=args.encoding,
                    errors=args.errors,
                )
            )
        emit(
            {
                "mode": "batch",
                "files": len(summaries),
                "total_written": sum(s["written"] for s in summaries),
                "total_skipped": sum(s["skipped"] for s in summaries),
                "per_file": summaries,
            }
        )
        return 0

    inp = args.input
    if inp is None:
        # Sensible default when someone runs the script with no args
        default_csv = _ROOT / "dataset" / "gaming_minecraft.csv"
        if default_csv.is_file():
            inp = default_csv
        else:
            p.error("Pass --input FILE.csv, --input-dir DIR, or --input DIR (directory of CSVs).")

    if inp.is_dir():
        raise SystemExit(
            f"{inp} is a directory. Use --input-dir {inp} or --output-dir for batch mode."
        )

    out = args.output
    if out is None:
        out = args.output_dir / f"{inp.stem}.jsonl"

    summary = convert_one_csv(
        inp,
        out,
        source_stem=inp.stem,
        encoding=args.encoding,
        errors=args.errors,
    )
    emit({"mode": "single", **summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
