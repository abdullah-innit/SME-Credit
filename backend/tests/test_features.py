"""Hand-calculated feature and point-in-time regression tests."""

from datetime import date, datetime, timedelta
from types import SimpleNamespace as Row

import pytest

from app.ml.features import derive_features
from app.ml.registry import FEATURES, MODEL_FEATURES, validate_registry


def fixture():
    as_of = date(2026, 10, 10)
    app = Row(amount_requested_pkr=1200, tenor_months=12)
    business = Row(
        incorporated_on=date(2024, 10, 1), has_ntn=True, employees=3, collateral_value_pkr=2400
    )
    txns = [
        Row(
            date=date(2026, 10, 1),
            amount_pkr=600,
            direction="inflow",
            channel="bank",
            counterparty_hash="a",
            balance_after=300,
        ),
        Row(
            date=date(2026, 10, 2),
            amount_pkr=200,
            direction="inflow",
            channel="cash_deposit",
            counterparty_hash="b",
            balance_after=500,
        ),
        Row(
            date=date(2026, 10, 3),
            amount_pkr=300,
            direction="outflow",
            channel="bank",
            counterparty_hash="c",
            balance_after=200,
        ),
    ]
    bills = [
        Row(due_date=date(2026, 10, 5), paid_date=date(2026, 10, 4)),
        Row(due_date=date(2026, 10, 7), paid_date=date(2026, 10, 9)),
    ]
    bureau = [
        Row(id=1, created_at=datetime(2026, 9, 1), score=680, delinquent_flag=False, active_loans=1)
    ]
    return as_of, app, business, txns, bills, bureau


def test_known_answers_and_scenarios():
    as_of, app, business, txns, bills, bureau = fixture()
    v = derive_features(1, app, business, txns, bills, bureau, as_of=as_of).values
    assert len(v) == 30
    assert v["avg_monthly_inflow"] == pytest.approx(800 / 6)
    assert v["avg_monthly_net_cash_flow"] == pytest.approx(500 / 6)
    assert v["dscr_proxy"] == pytest.approx((500 / 6) / 100)
    assert v["instalment_to_inflow"] == pytest.approx(0.75)
    assert v["top3_customer_concentration"] == 1
    assert v["digital_payment_share"] == 0.75
    assert v["cash_deposit_share"] == 0.25
    assert v["on_time_bill_ratio"] == 0.5
    assert v["avg_days_late"] == 1
    assert v["bureau_score"] == 680
    assert v["collateral_coverage"] == 2
    assert v["has_debt_service_data"] == 0
    changed = derive_features(
        1,
        app,
        business,
        txns,
        bills,
        bureau,
        as_of=as_of,
        amount_requested_pkr=600,
        tenor_months=6,
        balance_override_pkr=1000,
        digital_share_override=0.9,
        existing_debt_service_pkr=50,
    ).values
    assert changed["collateral_coverage"] == 4
    assert changed["avg_balance_ratio"] == pytest.approx(7.5)
    assert changed["digital_payment_share"] == 0.9
    assert changed["dscr_proxy"] == pytest.approx((500 / 6) / 150)
    assert changed["has_debt_service_data"] == 1


def test_no_look_ahead_and_missingness():
    as_of, app, business, txns, bills, bureau = fixture()
    base = derive_features(1, app, business, txns, bills, bureau, as_of=as_of).values
    future = Row(
        date=as_of + timedelta(days=1),
        amount_pkr=999999,
        direction="inflow",
        channel="bank",
        counterparty_hash="z",
        balance_after=999999,
    )
    same_day = Row(
        date=as_of,
        amount_pkr=999999,
        direction="inflow",
        channel="bank",
        counterparty_hash="z",
        balance_after=999999,
    )
    future_bill = Row(due_date=as_of + timedelta(days=1), paid_date=None)
    future_bureau = Row(
        id=2, created_at=datetime(2026, 10, 11), score=100, delinquent_flag=True, active_loans=9
    )
    actual = derive_features(
        1,
        app,
        business,
        txns + [same_day, future],
        bills + [future_bill],
        bureau + [future_bureau],
        as_of=as_of,
    ).values
    assert actual == base
    missing = derive_features(1, app, business, [], [], [], as_of=as_of).values
    assert missing["bureau_score"] == 600
    assert missing["has_bureau_record"] == 0
    assert missing["has_transaction_history"] == 0
    assert missing["has_bill_history"] == 0
    assert missing["has_balance_data"] == 0


def test_registry_complete_and_protected_fields_excluded():
    assert set(MODEL_FEATURES) == set(FEATURES)
    assert not any("gender" in key or "age_band" in key for key in FEATURES)
    with pytest.raises(ValueError, match="missing metadata"):
        validate_registry(MODEL_FEATURES + ("unregistered",))


def test_compute_persists_snapshot_without_committing_caller_session():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app.db.models import Application, FeatureSnapshot
    from app.db.seed import seed_demo
    from app.db.session import init_db, make_engine
    from app.ml.features import compute_features

    engine = make_engine("sqlite://")
    init_db(engine, seed_hook=seed_demo)
    with Session(engine) as session:
        application = session.scalar(select(Application))
        result = compute_features(application.id, session)
        snapshot = session.get(FeatureSnapshot, result.snapshot_id)
        assert snapshot.feature_version == result.feature_version
        assert snapshot.values == result.values
        assert result.values["has_transaction_history"] == 0
        session.rollback()
    with Session(engine) as session:
        assert session.scalar(select(FeatureSnapshot)) is None
    engine.dispose()


def test_bureau_at_submission_timestamp_is_available():
    as_of, app, business, txns, bills, _ = fixture()
    cutoff = datetime(2026, 10, 10, 9, 0)
    same = Row(id=2, created_at=cutoff, score=710, delinquent_flag=False, active_loans=1)
    later = Row(
        id=3,
        created_at=cutoff + timedelta(seconds=1),
        score=100,
        delinquent_flag=True,
        active_loans=9,
    )
    result = derive_features(
        1, app, business, txns, bills, [same, later], as_of=as_of, bureau_cutoff=cutoff
    ).values
    assert result["bureau_score"] == 710
    assert result["has_bureau_record"] == 1
