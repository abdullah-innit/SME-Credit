"""Relational credit records. Money is integer PKR; no raw banking identifiers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app import enums
from app.db.base import Base, TimestampMixin


def enum_type(cls: type) -> Enum:
    return Enum(
        cls,
        values_callable=lambda e: [v.value for v in e],
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[enums.UserRole] = mapped_column(
        enum_type(enums.UserRole), nullable=False, default=enums.UserRole.APPLICANT
    )

    businesses: Mapped[list[Business]] = relationship(
        back_populates="owner", foreign_keys="Business.owner_id"
    )
    consent_records: Mapped[list[ConsentRecord]] = relationship(
        back_populates="user", foreign_keys="ConsentRecord.user_id"
    )
    decisions: Mapped[list[Decision]] = relationship(
        back_populates="decided_by_user", foreign_keys="Decision.decided_by"
    )
    memo_records: Mapped[list[MemoRecord]] = relationship(
        back_populates="created_by_user", foreign_keys="MemoRecord.created_by"
    )
    ai_call_logs: Mapped[list[AiCallLog]] = relationship(
        back_populates="actor_user", foreign_keys="AiCallLog.actor"
    )
    audit_logs: Mapped[list[AuditLog]] = relationship(
        back_populates="actor_user", foreign_keys="AuditLog.actor"
    )


class Business(TimestampMixin, Base):
    __tablename__ = "businesses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(1024), nullable=False)
    sector: Mapped[str] = mapped_column(String(1024), nullable=False)
    city: Mapped[str] = mapped_column(String(1024), nullable=False)
    province: Mapped[str] = mapped_column(String(1024), nullable=False)
    legal_form: Mapped[enums.LegalForm] = mapped_column(enum_type(enums.LegalForm), nullable=False)
    incorporated_on: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    employees: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    owner_gender: Mapped[enums.Gender] = mapped_column(
        enum_type(enums.Gender), nullable=False, default=enums.Gender.UNDISCLOSED
    )
    owner_age_band: Mapped[enums.AgeBand] = mapped_column(
        enum_type(enums.AgeBand), nullable=False, default=enums.AgeBand.UNDISCLOSED
    )
    is_women_led: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_ntn: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_synthetic: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )
    collateral_value_pkr: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    __table_args__ = (
        CheckConstraint("employees >= 0", name="ck_businesses_employees"),
        CheckConstraint(
            "collateral_value_pkr >= 0 "
            "AND collateral_value_pkr = CAST(collateral_value_pkr AS BIGINT)",
            name="ck_businesses_collateral_value_pkr",
        ),
    )

    owner: Mapped[User] = relationship(back_populates="businesses", foreign_keys=[owner_id])
    consent_records: Mapped[list[ConsentRecord]] = relationship(
        back_populates="business", foreign_keys="ConsentRecord.business_id"
    )
    applications: Mapped[list[Application]] = relationship(
        back_populates="business", foreign_keys="Application.business_id"
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="business", foreign_keys="Document.business_id"
    )
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="business", foreign_keys="Transaction.business_id"
    )
    bill_payments: Mapped[list[BillPayment]] = relationship(
        back_populates="business", foreign_keys="BillPayment.business_id"
    )
    bureau_records: Mapped[list[BureauRecord]] = relationship(
        back_populates="business", foreign_keys="BureauRecord.business_id"
    )


class ConsentRecord(TimestampMixin, Base):
    __tablename__ = "consent_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    business_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("businesses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    scope: Mapped[str] = mapped_column(String(1024), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(1024), nullable=False)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    business: Mapped[Business] = relationship(
        back_populates="consent_records", foreign_keys=[business_id]
    )
    user: Mapped[User] = relationship(back_populates="consent_records", foreign_keys=[user_id])
    applications: Mapped[list[Application]] = relationship(
        back_populates="consent", foreign_keys="Application.consent_id"
    )


class Application(TimestampMixin, Base):
    __tablename__ = "applications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    business_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("businesses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    product: Mapped[enums.Product] = mapped_column(enum_type(enums.Product), nullable=False)
    amount_requested_pkr: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tenor_months: Mapped[int] = mapped_column(Integer, nullable=False)
    purpose: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[enums.ApplicationStatus] = mapped_column(
        enum_type(enums.ApplicationStatus), nullable=False, default=enums.ApplicationStatus.DRAFT
    )
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    consent_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("consent_records.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    __table_args__ = (
        CheckConstraint(
            "amount_requested_pkr >= 0 "
            "AND amount_requested_pkr = CAST(amount_requested_pkr AS BIGINT)",
            name="ck_applications_amount_requested_pkr",
        ),
        CheckConstraint("tenor_months > 0", name="ck_applications_tenor_months"),
    )

    business: Mapped[Business] = relationship(
        back_populates="applications", foreign_keys=[business_id]
    )
    consent: Mapped[ConsentRecord] = relationship(
        back_populates="applications", foreign_keys=[consent_id]
    )
    feature_snapshots: Mapped[list[FeatureSnapshot]] = relationship(
        back_populates="application", foreign_keys="FeatureSnapshot.application_id"
    )
    score_results: Mapped[list[ScoreResult]] = relationship(
        back_populates="application", foreign_keys="ScoreResult.application_id"
    )
    integrity_results: Mapped[list[IntegrityResult]] = relationship(
        back_populates="application", foreign_keys="IntegrityResult.application_id"
    )
    fraud_flags: Mapped[list[FraudFlag]] = relationship(
        back_populates="application", foreign_keys="FraudFlag.application_id"
    )
    decisions: Mapped[list[Decision]] = relationship(
        back_populates="application", foreign_keys="Decision.application_id"
    )
    memo_records: Mapped[list[MemoRecord]] = relationship(
        back_populates="application", foreign_keys="MemoRecord.application_id"
    )
    ai_call_logs: Mapped[list[AiCallLog]] = relationship(
        back_populates="application", foreign_keys="AiCallLog.application_id"
    )


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    business_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("businesses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    type: Mapped[enums.DocumentType] = mapped_column(enum_type(enums.DocumentType), nullable=False)
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    parse_status: Mapped[enums.ParseStatus] = mapped_column(
        enum_type(enums.ParseStatus), nullable=False, default=enums.ParseStatus.PENDING
    )
    parse_report: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    __table_args__ = (
        CheckConstraint(
            "length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'", name="ck_documents_sha256"
        ),
    )

    business: Mapped[Business] = relationship(
        back_populates="documents", foreign_keys=[business_id]
    )
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="document", foreign_keys="Transaction.document_id"
    )
    bill_payments: Mapped[list[BillPayment]] = relationship(
        back_populates="document", foreign_keys="BillPayment.document_id"
    )
    bureau_records: Mapped[list[BureauRecord]] = relationship(
        back_populates="document", foreign_keys="BureauRecord.document_id"
    )


class Transaction(TimestampMixin, Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    business_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("businesses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    amount_pkr: Mapped[int] = mapped_column(BigInteger, nullable=False)
    direction: Mapped[enums.Direction] = mapped_column(enum_type(enums.Direction), nullable=False)
    channel: Mapped[enums.Channel] = mapped_column(enum_type(enums.Channel), nullable=False)
    category: Mapped[str] = mapped_column(String(1024), nullable=False)
    counterparty_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    balance_after: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    document_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    __table_args__ = (
        CheckConstraint(
            "amount_pkr >= 0 AND amount_pkr = CAST(amount_pkr AS BIGINT)",
            name="ck_transactions_amount_pkr",
        ),
        CheckConstraint(
            "length(counterparty_hash) = 64 AND counterparty_hash NOT GLOB '*[^0-9a-f]*'",
            name="ck_transactions_counterparty_hash",
        ),
        CheckConstraint(
            "balance_after >= 0 AND balance_after = CAST(balance_after AS BIGINT)",
            name="ck_transactions_balance_after",
        ),
    )

    business: Mapped[Business] = relationship(
        back_populates="transactions", foreign_keys=[business_id]
    )
    document: Mapped[Document | None] = relationship(
        back_populates="transactions", foreign_keys=[document_id]
    )


class BillPayment(TimestampMixin, Base):
    __tablename__ = "bill_payments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    business_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("businesses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    utility_type: Mapped[enums.UtilityType] = mapped_column(
        enum_type(enums.UtilityType), nullable=False
    )
    due_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    paid_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    amount_pkr: Mapped[int] = mapped_column(BigInteger, nullable=False)
    __table_args__ = (
        CheckConstraint(
            "amount_pkr >= 0 AND amount_pkr = CAST(amount_pkr AS BIGINT)",
            name="ck_bill_payments_amount_pkr",
        ),
    )

    business: Mapped[Business] = relationship(
        back_populates="bill_payments", foreign_keys=[business_id]
    )
    document: Mapped[Document | None] = relationship(
        back_populates="bill_payments", foreign_keys=[document_id]
    )


class BureauRecord(TimestampMixin, Base):
    __tablename__ = "bureau_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    business_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("businesses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active_loans: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    delinquent_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    history_months: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    __table_args__ = (
        CheckConstraint("score >= 0", name="ck_bureau_records_score"),
        CheckConstraint("active_loans >= 0", name="ck_bureau_records_active_loans"),
        CheckConstraint("history_months >= 0", name="ck_bureau_records_history_months"),
    )

    business: Mapped[Business] = relationship(
        back_populates="bureau_records", foreign_keys=[business_id]
    )
    document: Mapped[Document | None] = relationship(
        back_populates="bureau_records", foreign_keys=[document_id]
    )


class FeatureSnapshot(TimestampMixin, Base):
    __tablename__ = "feature_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    feature_version: Mapped[str] = mapped_column(String(1024), nullable=False)
    values: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    application: Mapped[Application] = relationship(
        back_populates="feature_snapshots", foreign_keys=[application_id]
    )
    score_results: Mapped[list[ScoreResult]] = relationship(
        back_populates="feature_snapshot", foreign_keys="ScoreResult.feature_snapshot_id"
    )


class ScoreResult(TimestampMixin, Base):
    __tablename__ = "score_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    feature_snapshot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_snapshots.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(1024), nullable=False)
    pd: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    tier: Mapped[enums.RiskTier] = mapped_column(enum_type(enums.RiskTier), nullable=False)
    reasons: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    counterfactuals: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    __table_args__ = (
        CheckConstraint("pd >= 0 AND pd <= 1", name="ck_score_results_pd"),
        CheckConstraint("score >= 0", name="ck_score_results_score"),
    )

    application: Mapped[Application] = relationship(
        back_populates="score_results", foreign_keys=[application_id]
    )
    feature_snapshot: Mapped[FeatureSnapshot] = relationship(
        back_populates="score_results", foreign_keys=[feature_snapshot_id]
    )


class IntegrityResult(TimestampMixin, Base):
    __tablename__ = "integrity_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    severity: Mapped[enums.Severity] = mapped_column(enum_type(enums.Severity), nullable=False)
    code: Mapped[str] = mapped_column(String(1024), nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    application: Mapped[Application] = relationship(
        back_populates="integrity_results", foreign_keys=[application_id]
    )
    fraud_flags: Mapped[list[FraudFlag]] = relationship(
        back_populates="integrity_result", foreign_keys="FraudFlag.integrity_result_id"
    )


class FraudFlag(TimestampMixin, Base):
    __tablename__ = "fraud_flags"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    integrity_result_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("integrity_results.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    severity: Mapped[enums.Severity] = mapped_column(enum_type(enums.Severity), nullable=False)
    code: Mapped[str] = mapped_column(String(1024), nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    application: Mapped[Application] = relationship(
        back_populates="fraud_flags", foreign_keys=[application_id]
    )
    integrity_result: Mapped[IntegrityResult | None] = relationship(
        back_populates="fraud_flags", foreign_keys=[integrity_result_id]
    )


class Decision(TimestampMixin, Base):
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    outcome: Mapped[enums.Outcome] = mapped_column(enum_type(enums.Outcome), nullable=False)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tenor: Mapped[int] = mapped_column(Integer, nullable=False)
    rate_band: Mapped[str] = mapped_column(String(1024), nullable=False)
    decided_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    is_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    override_reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    policy_version: Mapped[str] = mapped_column(String(1024), nullable=False)
    __table_args__ = (
        CheckConstraint(
            "amount >= 0 AND amount = CAST(amount AS BIGINT)", name="ck_decisions_amount"
        ),
        CheckConstraint("tenor > 0", name="ck_decisions_tenor"),
        CheckConstraint(
            "NOT is_override OR (override_reason IS NOT NULL "
            "AND length(trim(override_reason)) > 0)",
            name="ck_decisions_override_reason",
        ),
    )

    application: Mapped[Application] = relationship(
        back_populates="decisions", foreign_keys=[application_id]
    )
    decided_by_user: Mapped[User] = relationship(
        back_populates="decisions", foreign_keys=[decided_by]
    )
    memo_records: Mapped[list[MemoRecord]] = relationship(
        back_populates="decision", foreign_keys="MemoRecord.decision_id"
    )


class MemoRecord(TimestampMixin, Base):
    __tablename__ = "memo_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    decision_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("decisions.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    created_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(1024), nullable=False)
    content: Mapped[str] = mapped_column(String(1024), nullable=False)

    application: Mapped[Application] = relationship(
        back_populates="memo_records", foreign_keys=[application_id]
    )
    decision: Mapped[Decision | None] = relationship(
        back_populates="memo_records", foreign_keys=[decision_id]
    )
    created_by_user: Mapped[User] = relationship(
        back_populates="memo_records", foreign_keys=[created_by]
    )


class AiCallLog(TimestampMixin, Base):
    __tablename__ = "ai_call_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("applications.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    actor: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(1024), nullable=False)
    model: Mapped[str] = mapped_column(String(1024), nullable=False)
    task: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[enums.AiCallStatus] = mapped_column(
        enum_type(enums.AiCallStatus), nullable=False
    )
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    output_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_version: Mapped[str] = mapped_column(String(1024), nullable=False)
    __table_args__ = (
        CheckConstraint(
            "length(input_hash) = 64 AND input_hash NOT GLOB '*[^0-9a-f]*'",
            name="ck_ai_call_logs_input_hash",
        ),
        CheckConstraint(
            "length(output_hash) = 64 AND output_hash NOT GLOB '*[^0-9a-f]*'",
            name="ck_ai_call_logs_output_hash",
        ),
        CheckConstraint("duration_ms >= 0", name="ck_ai_call_logs_duration_ms"),
    )

    application: Mapped[Application | None] = relationship(
        back_populates="ai_call_logs", foreign_keys=[application_id]
    )
    actor_user: Mapped[User | None] = relationship(
        back_populates="ai_call_logs", foreign_keys=[actor]
    )


class AuditLog(TimestampMixin, Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(1024), nullable=False)
    entity: Mapped[str] = mapped_column(String(1024), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    __table_args__ = (CheckConstraint("entity_id > 0", name="ck_audit_logs_entity_id"),)

    actor_user: Mapped[User | None] = relationship(
        back_populates="audit_logs", foreign_keys=[actor]
    )


@event.listens_for(AuditLog, "before_update")
@event.listens_for(AuditLog, "before_delete")
def prevent_audit_mutation(mapper: Any, connection: Any, target: AuditLog) -> None:
    raise ValueError("Audit records are append-only through the ORM")


@event.listens_for(Base, "before_insert", propagate=True)
@event.listens_for(Base, "before_update", propagate=True)
def validate_evidence(mapper: Any, connection: Any, target: Any) -> None:
    import re

    from app.schemas.base import validate_private_keys

    for column in mapper.columns:
        value = getattr(target, column.key)
        if column.key == "owner_age_band" and value is not None:
            value = enums.AgeBand(value)
            target.owner_age_band = value
        if isinstance(column.type, JSON):
            validate_private_keys(value)
        if (
            column.key in {"sha256", "counterparty_hash", "input_hash", "output_hash"}
            and value is not None
            and not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise ValueError("Identifiers must be lowercase SHA-256 digests")
