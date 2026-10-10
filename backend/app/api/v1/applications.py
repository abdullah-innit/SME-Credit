"""Application submission and typed scoring response."""

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import enums
from app.db.models import Application, AuditLog, Business, ConsentRecord
from app.db.session import get_db
from app.services.pipeline import score_application
from app.services.policy import PolicyCheck

router = APIRouter(tags=["applications"])


class ApplicationInput(BaseModel):
    business_id: int = Field(gt=0)
    consent_id: int = Field(gt=0)
    product: enums.Product
    amount_requested_pkr: int = Field(gt=0)
    tenor_months: int = Field(gt=0)
    purpose: str = Field(min_length=1, max_length=1024)


class ApplicationOutput(ApplicationInput):
    id: int
    status: enums.ApplicationStatus
    submitted_at: datetime


class Offer(BaseModel):
    recommended_amount_pkr: int
    tenor_months: int
    indicative_rate_band: str
    expected_loss_pkr: float


class DataQuality(BaseModel):
    completeness: float
    transaction_count: int
    known_debt_service: bool
    statement_quality_score: float | None = None
    statement_count: int = 0


class Assurance(BaseModel):
    calibration_method: str
    test_brier_score: float
    confidence: str
    model_agreement: bool
    pd_gap: float
    data_quality: DataQuality
    policy_summary: dict[str, int]


class IntegrityHook(BaseModel):
    status: str
    flags: list[str]
    application_id: int


class ScoringOutput(BaseModel):
    application_id: int
    score_result_id: int
    decision_id: int
    model_version: str
    policy_version: str
    recommendation_only: bool
    idempotent_replay: bool
    champion_pd: float
    challenger_pd: float
    score: int
    tier: Literal["A", "B", "C", "D", "E"]
    outcome: Literal["APPROVE", "REFER", "DECLINE"]
    reason_codes: list[str]
    policy_checks: list[PolicyCheck]
    offer: Offer
    integrity: IntegrityHook
    assurance: Assurance


@router.post("/applications", response_model=ApplicationOutput, status_code=201)
def create_application(
    body: ApplicationInput, session: Annotated[Session, Depends(get_db)]
) -> ApplicationOutput:
    business = session.get(Business, body.business_id)
    consent = session.get(ConsentRecord, body.consent_id)
    if business is None or consent is None or consent.business_id != body.business_id:
        raise HTTPException(422, "Business and matching consent are required")
    now = datetime.now(UTC)
    if consent.revoked_at is not None or consent.granted_at.replace(tzinfo=UTC) > now:
        raise HTTPException(422, "Active consent is required")
    app = Application(
        **body.model_dump(), status=enums.ApplicationStatus.SUBMITTED, submitted_at=now
    )
    session.add(app)
    session.flush()
    session.add(
        AuditLog(
            actor=business.owner_id,
            action="application_submitted",
            entity="applications",
            entity_id=app.id,
            before=None,
            after={
                "business_id": body.business_id,
                "amount_requested_pkr": body.amount_requested_pkr,
                "tenor_months": body.tenor_months,
            },
            at=now,
        )
    )
    session.commit()
    return ApplicationOutput.model_validate(
        {**body.model_dump(), "id": app.id, "status": app.status, "submitted_at": now}
    )


@router.post("/applications/{application_id}/score", response_model=ScoringOutput)
def score(application_id: int) -> ScoringOutput:
    try:
        return ScoringOutput.model_validate(score_application(application_id))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
