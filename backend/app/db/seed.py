"""Explicit, idempotent synthetic seed: python -m app.db.seed."""

from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Application, Business, ConsentRecord, User
from app.db.session import init_db
from app.enums import LegalForm, Product, UserRole


def seed_demo(session: Session) -> None:
    if session.scalar(select(Business).where(Business.name == "Synthetic Demo SME")):
        return
    user = User(role=UserRole.APPLICANT)
    session.add(user)
    session.flush()
    business = Business(
        owner_id=user.id,
        name="Synthetic Demo SME",
        sector="retail",
        city="Taxila",
        province="Punjab",
        legal_form=LegalForm.SOLE_PROPRIETORSHIP,
        incorporated_on=date(2020, 1, 1),
    )
    session.add(business)
    session.flush()
    consent = ConsentRecord(
        business_id=business.id,
        user_id=user.id,
        scope="synthetic credit assessment",
        policy_version="demo-v1",
        granted_at=datetime.now(UTC),
    )
    session.add(consent)
    session.flush()
    session.add(
        Application(
            business_id=business.id,
            consent_id=consent.id,
            product=Product.WORKING_CAPITAL,
            amount_requested_pkr=100000,
            tenor_months=12,
            purpose="Synthetic demo stock",
        )
    )


if __name__ == "__main__":
    init_db(seed_hook=seed_demo)
    print("Database ready; synthetic demo seeded.")
