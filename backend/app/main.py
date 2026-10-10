"""FastAPI application entry point."""

from fastapi import FastAPI
app = FastAPI()

import traceback
import os
import sys

try:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager
    from fastapi.middleware.cors import CORSMiddleware
    from app.api.v1 import v1_router
    from app.core.config import settings
    from app.core.logging import logger
    from app.db.session import init_db
    from app.services.pipeline import warm_models

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan: runs on startup and shutdown."""
        init_db()
        try:
            warm_models()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Scoring artifacts unavailable: %s", exc)
        logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
        logger.info("Offline mode: %s", settings.OFFLINE_MODE)
        logger.info("CORS origin: %s", settings.FRONTEND_ORIGIN)
        yield

    app.router.lifespan_context = lifespan

    if os.getenv("VERCEL") or os.getenv("VERCEL_ENV"):
        init_db()
        try:
            warm_models()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Scoring artifacts unavailable: %s", exc)

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────
    app.include_router(v1_router)

except Exception as e:
    err_msg = traceback.format_exc()
    @app.get("/api/v1/health")
    def health_error():
        return {"status": "error", "error": err_msg, "type": "import_error"}
    @app.get("/api/test")
    def test_error():
        return {"status": "error", "error": err_msg}
