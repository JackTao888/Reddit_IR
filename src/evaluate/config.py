"""Defaults shared across pool generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


DEFAULT_POOL_DEPTH = 20
DEFAULT_TOP_K = 10
DEFAULT_METRICS_AT: Tuple[int, ...] = (5, 10)
DEFAULT_SHUFFLE_SEED = 42


@dataclass
class EvalConfig:
    pool_depth: int = DEFAULT_POOL_DEPTH
    top_k: int = DEFAULT_TOP_K
    metrics_at: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_METRICS_AT)
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED

    def validate(self) -> None:
        if self.pool_depth <= 0:
            raise ValueError("pool_depth must be > 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.metrics_at:
            raise ValueError("metrics_at must be non-empty")
        if any(k <= 0 for k in self.metrics_at):
            raise ValueError("metrics_at values must all be > 0")
