"""Bilingual, waterfall-ready explanation endpoint."""

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.ml.explain import explain_application

router = APIRouter(tags=["explanations"])


@router.get("/applications/{application_id}/explanation")
def get_application_explanation(application_id: int) -> dict[str, Any]:
    if application_id <= 0:
        raise HTTPException(status_code=404, detail="Application not found")
    try:
        return asdict(explain_application(application_id))
    except ValueError as exc:
        if "does not exist" in str(exc):
            raise HTTPException(status_code=404, detail="Application not found") from exc
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SQLAlchemyError:
        raise
