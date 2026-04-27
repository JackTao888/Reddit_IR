"""Reddit crawler package for the JHU 666 IR final project.

Public entry points:
- CrawlerConfig: typed run configuration
- Runner: orchestrates a crawl
- main: CLI entry point (``python -m src.crawler.cli``)
"""

from .config import CrawlerConfig, load_env_credentials
from .runner import Runner

__all__ = ["CrawlerConfig", "load_env_credentials", "Runner"]
