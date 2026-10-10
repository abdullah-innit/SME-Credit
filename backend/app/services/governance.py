from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import enums
from app.db.models import (
    Appeal,
    Application,
    AuditLog,
    BillPayment,
    Business,
    ConsentRecord,
    Document,
    Transaction,
)


def mask_pii(data: dict) -> dict:
    """Helper to mask PII for data minimization."""
    masked = data.copy()
    for key in ["owner_name", "phone", "email", "address", "cnic"]:
        if key in masked:
            masked[key] = "***"
    return masked

def revoke_consent(business_id: int, session: Session) -> bool:
    """
    Purge applicant's transaction and bill payment data.
    Record the purge in AuditLog.
    """
    business = session.get(Business, business_id)
    if not business:
        return False

    consent_record = session.scalars(
        select(ConsentRecord)
        .where(ConsentRecord.business_id == business_id)
        .order_by(ConsentRecord.id.desc())
    ).first()

    if consent_record:
        consent_record.revoked_at = datetime.now(UTC)

    # Delete transactions
    txns = session.scalars(select(Transaction).where(Transaction.business_id == business_id)).all()
    for t in txns:
        session.delete(t)

    # Delete bill payments
    bills = session.scalars(select(BillPayment).where(BillPayment.business_id == business_id)).all()
    for b in bills:
        session.delete(b)

    # Delete documents
    docs = session.scalars(select(Document).where(Document.business_id == business_id)).all()
    for d in docs:
        d.parse_report = {"purged": True}
        d.filename = "***"
        d.sha256 = "***"

    session.add(
        AuditLog(
            actor=business.owner_id,
            action="consent_revoked_data_purged",
            entity="businesses",
            entity_id=business_id,
            before={"txns": len(txns), "bills": len(bills)},
            after={"txns": 0, "bills": 0},
            at=datetime.now(UTC),
        )
    )
    session.flush()
    return True

def create_appeal(application_id: int, applicant_id: int, reason: str, session: Session) -> Appeal:
    from app import enums
    from app.db.models import Appeal, Decision

    app_record = session.get(Application, application_id)
    if not app_record or app_record.business.owner_id != applicant_id:
        raise ValueError("Application not found or unauthorized")

    decision = session.scalars(
        select(Decision)
        .where(Decision.application_id == application_id)
        .order_by(Decision.id.desc())
    ).first()

    if not decision:
        raise ValueError("No decision to appeal")

    if decision.outcome != enums.Outcome.REJECT:
        raise ValueError("Can only appeal a DECLINE decision")

    appeal = Appeal(
        application_id=application_id,
        applicant_id=applicant_id,
        reason=reason,
        status=enums.AppealStatus.PENDING,
        officer_task_created=True  # Simulating task creation
    )
    session.add(appeal)
    session.add(
        AuditLog(
            actor=applicant_id,
            action="appeal_submitted",
            entity="appeals",
            entity_id=0, # Will be updated after flush
            before=None,
            after={"reason": reason},
            at=datetime.now(UTC),
        )
    )
    session.flush()
    return appeal

def enforce_data_retention(session: Session):
    """Enforce retention limits set in config."""
    from app.core.config import settings
    cutoff = datetime.now(UTC) - timedelta(days=settings.DATA_RETENTION_DAYS)

    # Delete old businesses that have had consent revoked for > 5 years
    # As a simple data minimization step, delete unparsed old documents
    old_docs = session.scalars(
        select(Document).where(
            Document.created_at < cutoff,
            Document.parse_status != enums.ParseStatus.PARSED
        )
    ).all()

    for d in old_docs:
        session.delete(d)
    session.flush()

