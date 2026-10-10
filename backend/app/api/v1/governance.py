"""Fairness, Governance, and Consent routes."""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AuditLog
from app.db.session import get_db
from app.ml.fairness import generate_fairness_report, get_fairness_metrics
from app.services.governance import create_appeal, revoke_consent

router = APIRouter(tags=["governance"])


class RevokeConsentRequest(BaseModel):
    business_id: int


class RevokeConsentResponse(BaseModel):
    status: str
    purged: bool


class AppealRequest(BaseModel):
    applicant_id: int
    reason: str


class AppealResponse(BaseModel):
    id: int
    status: str
    officer_task_created: bool


class AuditLogOutput(BaseModel):
    id: int
    actor: int
    action: str
    entity: str
    entity_id: int
    at: datetime
    before: dict[str, Any] | None
    after: dict[str, Any] | None


@router.post("/consent/revoke", response_model=RevokeConsentResponse)
def api_revoke_consent(
    body: RevokeConsentRequest, session: Annotated[Session, Depends(get_db)]
) -> RevokeConsentResponse:
    purged = revoke_consent(body.business_id, session)
    if not purged:
        raise HTTPException(404, "Business not found or no active consent")
    session.commit()
    return RevokeConsentResponse(status="revoked", purged=True)


@router.get("/audit/log", response_model=list[AuditLogOutput])
def get_audit_log(
    session: Annotated[Session, Depends(get_db)],
    actor_id: int | None = None,
    action: str | None = None,
    entity_id: int | None = None,
    limit: int = 100,
) -> list[AuditLogOutput]:
    query = select(AuditLog).order_by(AuditLog.at.desc())
    if actor_id is not None:
        query = query.where(AuditLog.actor == actor_id)
    if action is not None:
        query = query.where(AuditLog.action == action)
    if entity_id is not None:
        query = query.where(AuditLog.entity_id == entity_id)

    logs = session.scalars(query.limit(limit)).all()
    return [
        AuditLogOutput(
            id=log_entry.id,
            actor=log_entry.actor,
            action=log_entry.action,
            entity=log_entry.entity,
            entity_id=log_entry.entity_id,
            at=log_entry.at,
            before=log_entry.before,
            after=log_entry.after
        )
        for log_entry in logs
    ]


@router.get("/audit/fairness")
def api_get_fairness(session: Annotated[Session, Depends(get_db)]) -> dict:
    return get_fairness_metrics(session)


@router.get("/audit/fairness/report", response_class=PlainTextResponse)
def api_get_fairness_report(session: Annotated[Session, Depends(get_db)]) -> str:
    return generate_fairness_report(session)


@router.post("/applications/{application_id}/appeal", response_model=AppealResponse)
def api_create_appeal(
    application_id: int,
    body: AppealRequest,
    session: Annotated[Session, Depends(get_db)],
) -> AppealResponse:
    try:
        appeal = create_appeal(application_id, body.applicant_id, body.reason, session)
        session.commit()
        return AppealResponse(
            id=appeal.id,
            status=appeal.status.value,
            officer_task_created=appeal.officer_task_created
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
