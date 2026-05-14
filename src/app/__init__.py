"""Flask app for the JHU 666 IR final project.

Public API:
- AppConfig: typed startup configuration
- create_app: Flask app factory (loads index + ranker pool once)
- RankerPool: lazy-init container of TfidfRanker / Bm25Ranker / Bm25 field-aware
- main: CLI entry point (``python -m src.app.cli``)
"""

from .app_factory import create_app
from .config import AppConfig
from .rankers_pool import RankerPool

__all__ = ["AppConfig", "create_app", "RankerPool"]
