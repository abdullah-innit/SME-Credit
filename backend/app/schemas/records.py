"""Create/read contracts for all shared records; ORM-compatible read models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated

from pydantic import Field, JsonValue, model_validator

from app import enums
from app.schemas.base import Contract, ReadContract

Money = Annotated[int, Field(strict=True, ge=0, le=9223372036854775807)]
Count = Annotated[int, Field(strict=True, ge=0)]
Positive = Annotated[int, Field(strict=True, gt=0)]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class UserCreate(Contract):
    role: enums.UserRole = enums.UserRole.APPLICANT


class UserRead(UserCreate, ReadContract):
    """Persisted User including identity and timestamps."""


class BusinessCreate(Contract):
    owner_id: Positive
    name: Text
    sector: Text
    city: Text
    province: Text
    legal_form: enums.LegalForm
    incorporated_on: date
    employees: Count = 0
    owner_gender: enums.Gender = enums.Gender.UNDISCLOSED
    owner_age_band: enums.AgeBand = enums.AgeBand.UNDISCLOSED
    is_women_led: bool = False
    has_ntn: bool = False
    is_synthetic: bool = True
    collateral_value_pkr: Money = 0


class BusinessRead(BusinessCreate, ReadContract):
    """Persisted Business including identity and timestamps."""


class ConsentRecordCreate(Contract):
    business_id: Positive
    user_id: Positive
    scope: Text
    policy_version: Text
    granted_at: datetime
    revoked_at: datetime | None = None


class ConsentRecordRead(ConsentRecordCreate, ReadContract):
    """Persisted ConsentRecord including identity and timestamps."""


class ApplicationCreate(Contract):
    business_id: Positive
    product: enums.Product
    amount_requested_pkr: Money
    tenor_months: Positive
    purpose: Text
    status: enums.ApplicationStatus = enums.ApplicationStatus.DRAFT
    submitted_at: datetime | None = None
    consent_id: Positive


class ApplicationRead(ApplicationCreate, ReadContract):
    """Persisted Application including identity and timestamps."""


class DocumentCreate(Contract):
    business_id: Positive
    type: enums.DocumentType
    filename: Text
    sha256: Hash
    parse_status: enums.ParseStatus = enums.ParseStatus.PENDING
    parse_report: dict[str, JsonValue] = Field(default_factory=dict)


class DocumentRead(DocumentCreate, ReadContract):
    """Persisted Document including identity and timestamps."""


class TransactionCreate(Contract):
    business_id: Positive
    date: date
    amount_pkr: Money
    direction: enums.Direction
    channel: enums.Channel
    category: Text
    counterparty_hash: Hash
    balance_after: Money | None = None
    document_id: Positive | None = None


class TransactionRead(TransactionCreate, ReadContract):
    """Persisted Transaction including identity and timestamps."""


class BillPaymentCreate(Contract):
    business_id: Positive
    document_id: Positive | None = None
    utility_type: enums.UtilityType
    due_date: date
    paid_date: date | None = None
    amount_pkr: Money


class BillPaymentRead(BillPaymentCreate, ReadContract):
    """Persisted BillPayment including identity and timestamps."""


class BureauRecordCreate(Contract):
    business_id: Positive
    document_id: Positive | None = None
    score: Count | None = None
    active_loans: Count = 0
    delinquent_flag: bool = False
    history_months: Count = 0


class BureauRecordRead(BureauRecordCreate, ReadContract):
    """Persisted BureauRecord including identity and timestamps."""


class FeatureSnapshotCreate(Contract):
    application_id: Positive
    feature_version: Text
    values: dict[str, JsonValue] = Field(default_factory=dict)


class FeatureSnapshotRead(FeatureSnapshotCreate, ReadContract):
    """Persisted FeatureSnapshot including identity and timestamps."""


class ScoreResultCreate(Contract):
    application_id: Positive
    feature_snapshot_id: Positive
    model_version: Text
    pd: Probability
    score: Count
    tier: enums.RiskTier
    reasons: list[JsonValue] = Field(default_factory=list)
    counterfactuals: list[JsonValue] = Field(default_factory=list)


class ScoreResultRead(ScoreResultCreate, ReadContract):
    """Persisted ScoreResult including identity and timestamps."""


class IntegrityResultCreate(Contract):
    application_id: Positive
    severity: enums.Severity
    code: Text
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class IntegrityResultRead(IntegrityResultCreate, ReadContract):
    """Persisted IntegrityResult including identity and timestamps."""


class FraudFlagCreate(Contract):
    application_id: Positive
    integrity_result_id: Positive | None = None
    severity: enums.Severity
    code: Text
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class FraudFlagRead(FraudFlagCreate, ReadContract):
    """Persisted FraudFlag including identity and timestamps."""


class DecisionCreate(Contract):
    application_id: Positive
    outcome: enums.Outcome
    amount: Money
    tenor: Positive
    rate_band: Text
    decided_by: Positive
    is_override: bool = False
    override_reason: Text | None = None
    policy_version: Text

    @model_validator(mode="after")
    def override_requires_reason(self) -> DecisionCreate:
        if self.is_override and not (self.override_reason and self.override_reason.strip()):
            raise ValueError("An override requires a reason")
        return self


class DecisionRead(DecisionCreate, ReadContract):
    """Persisted Decision including identity and timestamps."""


class MemoRecordCreate(Contract):
    application_id: Positive
    decision_id: Positive | None = None
    created_by: Positive
    version: Text
    content: Text


class MemoRecordRead(MemoRecordCreate, ReadContract):
    """Persisted MemoRecord including identity and timestamps."""


class AiCallLogCreate(Contract):
    application_id: Positive | None = None
    actor: Positive | None = None
    provider: Text
    model: Text
    task: Text
    status: enums.AiCallStatus
    input_hash: Hash
    output_hash: Hash | None = None
    duration_ms: Count = 0
    prompt_version: Text


class AiCallLogRead(AiCallLogCreate, ReadContract):
    """Persisted AiCallLog including identity and timestamps."""


class AuditLogCreate(Contract):
    actor: Positive | None = None
    action: Text
    entity: Text
    entity_id: Positive
    before: dict[str, JsonValue] | None = None
    after: dict[str, JsonValue] | None = None
    at: datetime


class AuditLogRead(AuditLogCreate, ReadContract):
    """Persisted AuditLog including identity and timestamps."""
