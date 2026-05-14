"""Typed startup configuration for the Flask app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_RANKERS = ("tfidf", "bm25", "bm25_field")


@dataclass
class AppConfig:
    index_dir: Path = Path("data/index")
    default_ranker: str = "bm25"
    default_top_k: int = 10
    max_top_k: int = 50

    def __post_init__(self) -> None:
        self.index_dir = Path(self.index_dir)

    def validate(self) -> None:
        if self.default_ranker not in VALID_RANKERS:
            raise ValueError(
                f"default_ranker must be one of {VALID_RANKERS}"
            )
        if self.default_top_k <= 0:
            raise ValueError("default_top_k must be > 0")
        if self.max_top_k < self.default_top_k:
            raise ValueError("max_top_k must be >= default_top_k")
