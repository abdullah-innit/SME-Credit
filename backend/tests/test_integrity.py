"""Tests for the integrity engine rules."""

from datetime import date, timedelta

from app import enums
from app.db.models import Transaction
from app.ml.integrity import evaluate_rules


class MockBusiness:
    pass


class MockApplication:
    pass


class MockSession:
    def get(self, *args, **kwargs):
        return None

    def scalars(self, *args, **kwargs):
        class MockResult:
            def all(self):
                return []

        return MockResult()


def create_txn(id, amount, direction, balance, dt, counterparty="A"):
    t = Transaction()
    t.id = id
    t.amount_pkr = amount
    t.direction = enums.Direction(direction)
    t.balance_after = balance
    t.date = dt
    t.counterparty_hash = counterparty
    return t


def test_balance_reconciliation():
    d = date(2025, 1, 1)
    txns = [
        create_txn(1, 100, "inflow", 100, d),
        create_txn(2, 50, "inflow", 200, d),  # break here, expected 150
        create_txn(3, 50, "outflow", 150, d),
    ]
    flags = evaluate_rules(1, 1, txns, date(2025, 1, 2), MockSession(), 500)
    assert any(f["code"] == "RECONCILIATION_BREAK" for f in flags)


def test_window_dressing():
    d_prior = date(2025, 1, 1)
    d_window = date(2025, 3, 1)
    as_of = date(2025, 3, 10)
    txns = [
        create_txn(1, 100, "inflow", 100, d_prior),
        create_txn(2, 100, "inflow", 200, d_prior),
        create_txn(3, 1000, "inflow", 1200, d_window),  # spike
        create_txn(4, 900, "outflow", 300, d_window + timedelta(days=1)),  # withdrawal
    ]
    flags = evaluate_rules(1, 1, txns, as_of, MockSession(), 500)
    assert any(f["code"] == "WINDOW_DRESSING" for f in flags)


def test_circular_flows():
    d = date(2025, 1, 1)
    txns = [
        create_txn(1, 1000, "inflow", 1000, d, counterparty="HASH1"),
        create_txn(2, 1000, "outflow", 0, d + timedelta(days=1), counterparty="HASH1"),
    ]
    flags = evaluate_rules(1, 1, txns, date(2025, 1, 5), MockSession(), 500)
    assert any(f["code"] == "CIRCULAR_FLOWS" for f in flags)


def test_round_number():
    d = date(2025, 1, 1)
    txns = [
        create_txn(1, 10000, "inflow", 10000, d),
        create_txn(2, 50000, "inflow", 60000, d),
        create_txn(3, 10000, "inflow", 70000, d),
    ]
    flags = evaluate_rules(1, 1, txns, date(2025, 1, 2), MockSession(), 500)
    assert any(f["code"] == "ROUND_NUMBER_SHARE" for f in flags)


def test_benford():
    d = date(2025, 1, 1)
    # create all numbers starting with 9 to trigger benford
    txns = [create_txn(i, 900 + i, "inflow", None, d) for i in range(1, 25)]
    flags = evaluate_rules(1, 1, txns, date(2025, 1, 2), MockSession(), 500)
    assert any(f["code"] == "BENFORD_DEVIATION" for f in flags)


def test_timestamp_anomaly():
    d = date(2050, 1, 1)  # impossible date
    txns = [create_txn(1, 100, "inflow", 100, d)]
    flags = evaluate_rules(1, 1, txns, date(2025, 1, 2), MockSession(), 500)
    assert any(f["code"] == "TIMESTAMP_ANOMALY" for f in flags)


def test_revenue_mismatch():
    d = date(2025, 1, 1)
    txns = [create_txn(1, 100, "inflow", 100, d)]
    flags = evaluate_rules(1, 1, txns, date(2025, 2, 1), MockSession(), 100000)  # massive request
    assert any(f["code"] == "REVENUE_MISMATCH" for f in flags)
