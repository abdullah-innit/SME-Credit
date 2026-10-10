"""Credit Memo and Plain-Language Explanation Routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import Application
from app.db.session import get_db
from app.services.memo import (
    get_or_create_memo,
    list_memo_versions,
    render_memo_pdf,
    render_print_html,
    save_edited_memo,
)

router = APIRouter(tags=["credit-memo"])


class MemoUpdateRequest(BaseModel):
    memo_data: dict[str, Any]
    user_id: int | None = Field(default=None, description="Officer user ID")


class MemoRegenerateRequest(BaseModel):
    tone: str = Field(default="detailed", description="'concise' or 'detailed'")
    user_id: int | None = Field(default=None, description="Requesting officer ID")


@router.get("/applications/{application_id}/memo")
def get_memo(
    application_id: int,
    session: Annotated[Session, Depends(get_db)],
    tone: str = Query(default="detailed", pattern="^(concise|detailed)$"),
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """Retrieve existing credit memo or generate a new version."""
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    try:
        return get_or_create_memo(
            application_id,
            session,
            tone=tone,
            force_regenerate=force_regenerate,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate credit memo: {exc}"
        ) from exc


@router.post("/applications/{application_id}/memo/generate")
def regenerate_memo(
    application_id: int,
    body: MemoRegenerateRequest,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Regenerate credit memo with specified tone (concise or detailed) and store as new version."""
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    try:
        return get_or_create_memo(
            application_id,
            session,
            tone=body.tone,
            user_id=body.user_id,
            force_regenerate=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Regeneration error: {exc}") from exc


@router.put("/applications/{application_id}/memo")
def edit_memo(
    application_id: int,
    body: MemoUpdateRequest,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Officer edit: persist updated memo content as a new version with unified diff."""
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    try:
        return save_edited_memo(
            application_id,
            body.memo_data,
            session,
            user_id=body.user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save memo edit: {exc}") from exc


@router.get("/applications/{application_id}/memo/versions")
def get_memo_versions(
    application_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """List historical memo versions and diff status for an application."""
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    return list_memo_versions(application_id, session)


@router.get("/applications/{application_id}/memo/export/html", response_class=HTMLResponse)
def export_memo_html(
    application_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    """Export print-optimised bilingual HTML view with Urdu RTL styling."""
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    memo = get_or_create_memo(application_id, session)
    html_content = render_print_html(memo)
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/applications/{application_id}/memo/export/pdf")
def export_memo_pdf(
    application_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> Response:
    """Export print-ready PDF credit memo generated via ReportLab."""
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    memo = get_or_create_memo(application_id, session)
    pdf_bytes = render_memo_pdf(memo)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=credit_memo_app_{application_id}.pdf"},
    )


@router.get("/applications/{application_id}/explanation/applicant")
def get_applicant_explanation(
    application_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Retrieve plain-language applicant explanation (English, Urdu, and Roman Urdu)."""
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    memo = get_or_create_memo(application_id, session)
    return memo.get("applicant_explanation", {})
