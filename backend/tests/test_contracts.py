"""Relational, validation and audit acceptance tests."""

from datetime import UTC, date, datetime
from enum import EnumType
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, configure_mappers

from app import enums
from app.db.base import Base
from app.db.models import Application, AuditLog, Business, Document, Transaction
from app.db.seed import seed_demo
from app.db.session import init_db, make_engine
from app.enums import Channel, Direction, DocumentType, Product
from app.schemas import records


@pytest.fixture
def engine():
    bind = make_engine("sqlite://")
    init_db(bind, seed_hook=seed_demo)
    yield bind
    bind.dispose()


def test_round_trip(engine):
    configure_mappers()
    with Session(engine) as session:
        business = session.scalar(select(Business))
        application = session.scalar(select(Application))
        session.add_all(
            [
                Transaction(
                    business_id=business.id,
                    date=date(2026, 10, 9),
                    amount_pkr=1000 + i,
                    direction=Direction.INFLOW,
                    channel=Channel.BANK,
                    category="sales",
                    counterparty_hash="a" * 64,
                )
                for i in range(100)
            ]
        )
        session.commit()
        ids = business.id, application.id
    with Session(engine) as session:
        business = session.get(Business, ids[0])
        assert len(business.transactions) == 100
        assert session.get(Application, ids[1]).business_id == business.id
        assert records.BusinessRead.model_validate(business).collateral_value_pkr == 0
        txn = session.scalar(select(Transaction))
        assert records.TransactionRead.model_validate(txn).amount_pkr == 1000
        assert isinstance(txn.amount_pkr, int)
        assert txn.created_at and txn.updated_at


@pytest.mark.parametrize("amount", [-1, 1.5, True, "100"])
def test_invalid_money(amount):
    with pytest.raises(ValidationError):
        records.ApplicationCreate(
            business_id=1,
            consent_id=1,
            product="working_capital",
            amount_requested_pkr=amount,
            tenor_months=12,
            purpose="stock",
        )


def test_all_enums():
    for enum in vars(enums).values():
        if isinstance(enum, EnumType) and enum.__module__ == enums.__name__:
            with pytest.raises(ValidationError):
                TypeAdapter(enum).validate_python("invalid")


def test_privacy_and_overrides():
    with pytest.raises(ValidationError):
        records.DocumentCreate(
            business_id=1,
            type="other",
            filename="synthetic",
            sha256="a" * 64,
            parse_report={"nested": [{"iban": "raw"}]},
        )
    with pytest.raises(ValidationError):
        records.UserCreate(role="applicant", account_number="raw")
    with pytest.raises(ValidationError):
        records.DecisionCreate(
            application_id=1,
            outcome="approve",
            amount=100,
            tenor=12,
            rate_band="demo",
            decided_by=1,
            is_override=True,
            policy_version="v1",
        )


def test_database_constraints(engine):
    with Session(engine) as session:
        session.add(
            Application(
                business_id=999,
                consent_id=999,
                product=Product.WORKING_CAPITAL,
                amount_requested_pkr=10,
                tenor_months=12,
                purpose="demo",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        application = session.scalar(select(Application))
        application.amount_requested_pkr = -1
        with pytest.raises(IntegrityError):
            session.commit()


def test_json_privacy(engine):
    with Session(engine) as session:
        business = session.scalar(select(Business))
        session.add(
            Document(
                business_id=business.id,
                type=DocumentType.OTHER,
                filename="demo",
                sha256="a" * 64,
                parse_report={"counterparty_name": "raw"},
            )
        )
        with pytest.raises(ValueError, match="identifiers"):
            session.commit()


def test_audit_and_seed(engine):
    init_db(engine, seed_hook=seed_demo)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Business)) == 1
        log = AuditLog(
            actor=None,
            action="create",
            entity="business",
            entity_id=1,
            before=None,
            after={"synthetic": True},
            at=datetime.now(UTC),
        )
        session.add(log)
        session.commit()
        log_id = log.id
        log.action = "tamper"
        with pytest.raises(ValueError, match="append-only"):
            session.commit()
        session.rollback()
        session.delete(session.get(AuditLog, log_id))
        with pytest.raises(ValueError, match="append-only"):
            session.commit()


def test_erd_and_indexes():
    erd = (Path(__file__).resolve().parents[2] / "docs/ERD.md").read_text()
    assert len(Base.metadata.tables) == 16
    for mapper in Base.registry.mappers:
        assert f"    {mapper.class_.__name__} {{" in erd
        for column in mapper.columns:
            assert f" {column.name}" in erd
            if column.name in {"business_id", "application_id", "date"}:
                assert column.index
            for fk in column.foreign_keys:
                assert f'{mapper.class_.__name__} : "{column.name}"' in erd
                assert fk.column.table in Base.metadata.sorted_tables


def test_all_money_fields_reject_negative_values():
    for contract in vars(records).values():
        if isinstance(contract, type) and contract.__name__.endswith("Create"):
            for name, field in contract.model_fields.items():
                if any(getattr(item, "ge", None) == 0 for item in field.metadata):
                    with pytest.raises(ValidationError) as error:
                        contract.model_validate({name: -1})
                    assert any(e["loc"] == (name,) for e in error.value.errors())


def test_seed_failure_rolls_back(engine):
    def failing_hook(session):
        from app.db.models import User
        from app.enums import UserRole

        session.add(User(role=UserRole.ADMIN))
        session.flush()
        raise RuntimeError("seed failed")

    from app.db.models import User

    with pytest.raises(RuntimeError):
        init_db(engine, seed_hook=failing_hook)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(User)) == 1


def test_database_rejects_fractional_money(engine):
    with Session(engine) as session:
        application = session.scalar(select(Application))
        application.amount_requested_pkr = 12.5
        with pytest.raises(IntegrityError):
            session.commit()


def test_erd_exact_columns_and_foreign_keys():
    import re

    erd = (Path(__file__).resolve().parents[2] / "docs/ERD.md").read_text()
    entities = dict(re.findall(r"    (\w+) \{\n(.*?)\n    \}", erd, re.S))
    assert set(entities) == {m.class_.__name__ for m in Base.registry.mappers}
    expected_edges = set()
    for mapper in Base.registry.mappers:
        columns = {line.split()[1] for line in entities[mapper.class_.__name__].splitlines()}
        assert columns == set(mapper.columns.keys())
        for column in mapper.columns:
            for fk in column.foreign_keys:
                parent = next(
                    m.class_.__name__
                    for m in Base.registry.mappers
                    if m.local_table is fk.column.table
                )
                expected_edges.add((parent, mapper.class_.__name__, column.name))
    actual_edges = set(re.findall(r'    (\w+) [|o]+--o\{ (\w+) : "(\w+)"', erd))
    assert actual_edges == expected_edges
