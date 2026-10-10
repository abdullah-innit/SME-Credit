"""Regressions for the reported synthetic flag and age-band mismatches."""

from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from app.db.migrate_synthetic import upgrade
from app.db.models import Business
from app.db.seed import seed_demo
from app.db.session import init_db, make_engine
from app.enums import AgeBand, OwnerAgeBand
from app.schemas.records import BusinessCreate, BusinessRead


def test_synthetic_and_legacy_age_round_trip():
    engine = make_engine("sqlite://")
    try:
        init_db(engine, seed_hook=seed_demo)
        with Session(engine) as session:
            business = session.scalar(select(Business))
            assert business.is_synthetic is True
            business.owner_age_band = "35-44"
            session.commit()
            session.expire_all()
            business = session.scalar(select(Business))
            assert business.owner_age_band == AgeBand.AGE_35_44
            data = BusinessRead.model_validate(business).model_dump()
            for name in ["id", "created_at", "updated_at"]:
                del data[name]
            data["owner_age_band"] = "35-44"
            contract = BusinessCreate.model_validate(data)
            assert contract.owner_age_band == OwnerAgeBand.BAND_35_44
            assert contract.model_dump(mode="json")["owner_age_band"] == "35_44"
            assert contract.is_synthetic is True
    finally:
        engine.dispose()


def test_upgrade_preserves_existing_data_and_is_idempotent():
    engine = make_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE businesses (id INTEGER PRIMARY KEY, name TEXT)"))
            connection.execute(text("INSERT INTO businesses VALUES (1, 'Synthetic example')"))
        assert upgrade(engine) is True
        assert upgrade(engine) is False
        assert "is_synthetic" in {c["name"] for c in inspect(engine).get_columns("businesses")}
        with engine.connect() as connection:
            row = connection.execute(text("SELECT name, is_synthetic FROM businesses")).one()
            assert tuple(row) == ("Synthetic example", 1)
    finally:
        engine.dispose()
