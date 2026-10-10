"""Health-check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

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
