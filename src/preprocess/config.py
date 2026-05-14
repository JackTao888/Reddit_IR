"""Typed configuration for the preprocessor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


VALID_STEMMERS = {"snowball", "porter", "none"}


@dataclass
class PreprocessConfig:
    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")

    lowercase: bool = True
    remove_stopwords: bool = True
    stem: bool = True
    stemmer: str = "snowball"

    min_token_length: int = 2
    drop_pure_numeric: bool = True

    strip_urls: bool = True
    strip_markdown: bool = True
    keep_subreddit_refs: bool = False

    selftext_excerpt_chars: int = 500

    extra_stopwords_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.extra_stopwords_path is not None:
            self.extra_stopwords_path = Path(self.extra_stopwords_path)

    def validate(self) -> None:
        if self.stemmer not in VALID_STEMMERS:
            raise ValueError(f"stemmer must be one of {sorted(VALID_STEMMERS)}")
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")
        if self.selftext_excerpt_chars < 0:
            raise ValueError("selftext_excerpt_chars must be >= 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_dir"] = str(self.input_dir)
        d["output_dir"] = str(self.output_dir)
        if self.extra_stopwords_path is not None:
            d["extra_stopwords_path"] = str(self.extra_stopwords_path)
        return d

    def preprocessing_metadata(self) -> dict:
        """Subset of options recorded inside each processed document.

        Captured fields fully describe how to reproduce tokenization at query
        time, so the ranker can match query tokens to indexed doc tokens.
        """
        effective_stem = self.stem and self.stemmer != "none"
        return {
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords,
            "stem": effective_stem,
            "stemmer": self.stemmer if effective_stem else "none",
            "min_token_length": self.min_token_length,
            "drop_pure_numeric": self.drop_pure_numeric,
            "strip_urls": self.strip_urls,
            "strip_markdown": self.strip_markdown,
            "keep_subreddit_refs": self.keep_subreddit_refs,
        }
