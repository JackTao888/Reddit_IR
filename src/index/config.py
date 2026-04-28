"""Typed configuration for the index builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


# Field names align with M2 preprocessor output: title_tokens, body_tokens,
# comments_tokens, all_tokens. The "all" field is required (used by plain
# BM25 and TF-IDF); per-field indexes enable field-aware BM25 in M5.
DEFAULT_FIELDS: List[str] = ["all", "title", "body", "comments"]


@dataclass
class IndexConfig:
    input_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/index")
    fields: List[str] = field(default_factory=lambda: list(DEFAULT_FIELDS))

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

    def validate(self) -> None:
        if not self.fields:
            raise ValueError("fields must be a non-empty list")
        for f in self.fields:
            if f not in {"all", "title", "body", "comments"}:
                raise ValueError(f"Unknown field: {f}")
        if "all" not in self.fields:
            raise ValueError("'all' field is required for ranking")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_dir"] = str(self.input_dir)
        d["output_dir"] = str(self.output_dir)
        return d
