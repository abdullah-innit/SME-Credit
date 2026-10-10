"""Health-check endpoint."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.llm.gateway import get_health_info

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Typed health-check response."""

    status: str
    version: str
    offline_mode: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return basic app health info."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        offline_mode=settings.OFFLINE_MODE,
    )


@router.get("/health/llm")
async def llm_health_check() -> dict[str, Any]:
    """Return LLM service layer health info (provider, mode, and last success)."""
    return get_health_info()
