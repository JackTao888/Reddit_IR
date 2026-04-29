"""Flask app factory: load index + rankers once, wire up routes."""

from __future__ import annotations

import logging

from flask import Flask

from ..index.store import load_artifacts
from .config import AppConfig
from .rankers_pool import RankerPool


log = logging.getLogger(__name__)


def create_app(app_config: AppConfig) -> Flask:
    app_config.validate()

    app = Flask(
        "src.app",
        template_folder="templates",
        static_folder="static",
    )

    artifacts = load_artifacts(app_config.index_dir)
    pool = RankerPool(artifacts)

    app.config["APP_CONFIG"] = app_config
    app.config["INDEX_ARTIFACTS"] = artifacts
    app.config["RANKER_POOL"] = pool

    @app.context_processor
    def _inject_globals():  # type: ignore[unused-ignore]
        return {
            "n_docs": len(artifacts.doc_store),
            "available_rankers": pool.available,
            "default_ranker": app_config.default_ranker,
        }

    from .views import main as main_blueprint

    app.register_blueprint(main_blueprint)

    log.info(
        "Flask app ready: n_docs=%d default_ranker=%s",
        len(artifacts.doc_store),
        app_config.default_ranker,
    )
    return app
