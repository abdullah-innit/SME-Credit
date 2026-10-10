"""API v1 router – aggregates all v1 endpoint routers."""

from fastapi import APIRouter

from app.api.v1.admin import router as admin_router
from app.api.v1.advisor import router as advisor_router
from app.api.v1.applications import router as applications_router
from app.api.v1.counterfactuals import router as counterfactuals_router
from app.api.v1.documents import router as documents_router
from app.api.v1.explanations import router as explanations_router
from app.api.v1.governance import router as governance_router
from app.api.v1.health import router as health_router
from app.api.v1.memo import router as memo_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health_router)
v1_router.include_router(applications_router)
v1_router.include_router(explanations_router)
v1_router.include_router(counterfactuals_router)
v1_router.include_router(documents_router)
v1_router.include_router(admin_router)
v1_router.include_router(memo_router)
v1_router.include_router(advisor_router)
v1_router.include_router(governance_router)
