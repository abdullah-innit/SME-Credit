"""API v1 router – aggregates all v1 endpoint routers."""

from fastapi import APIRouter

from app.api.v1.health import router as health_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health_router)
