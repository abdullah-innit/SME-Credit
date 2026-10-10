"""Point-in-time feature engineering for synthetic SME applications.

Six calendar months ending on the application date are used. Transactions on the
application date are excluded because their within-day ordering is unknown.
Missing continuous observations become zero, except bureau score (neutral 600).
Presence flags distinguish imputation from genuine zeros. The schema has no
existing debt instalment amount: DSCR assumes zero *known* existing service and
sets has_debt_service_data=0. It must not be presented as full debt coverage.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Application,
    BillPayment,
    BureauRecord,
    Business,
    FeatureSnapshot,
    Transaction,
)
from app.db.session import SessionLocal
from app.ml.registry import FEATURE_VERSION, MODEL_FEATURES, validate_registry


@dataclass(frozen=True)
class FeatureResult:
    application_id: int
    feature_version: str
    as_of: date
    values: dict[str, float]
    snapshot_id: int | None = None


def _month_shift(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _value(row: Any, name: str) -> Any:
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def _day(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))


def _divide(top: float, bottom: float) -> float:
    return top / bottom if bottom > 0 else 0.0


def derive_features(
    application_id: int,
    application: Any,
    business: Any,
    transactions: Iterable[Any],
    bills: Iterable[Any] = (),
    bureau_records: Iterable[Any] = (),
    *,
    as_of: date,
    bureau_cutoff: datetime | None = None,
    amount_requested_pkr: int | None = None,
    tenor_months: int | None = None,
    balance_override_pkr: float | None = None,
    digital_share_override: float | None = None,
    existing_debt_service_pkr: float | None = None,
) -> FeatureResult:
    """Pure recomputation entry point for counterfactuals; never mutates evidence."""
    validate_registry()
    if amount_requested_pkr is None:
        amount_requested_pkr = int(_value(application, "amount_requested_pkr"))
    if tenor_months is None:
        tenor_months = int(_value(application, "tenor_months"))
    if amount_requested_pkr < 0 or tenor_months <= 0:
        raise ValueError("Amount must be nonnegative and tenor positive")
    if balance_override_pkr is not None and balance_override_pkr < 0:
        raise ValueError("Balance override must be nonnegative")
    if digital_share_override is not None and not 0 <= digital_share_override <= 1:
        raise ValueError("Digital share must be between zero and one")
    if existing_debt_service_pkr is not None and existing_debt_service_pkr < 0:
        raise ValueError("Debt service must be nonnegative")

    first_month = _month_shift(as_of, -5)
    start = max(first_month, _day(_value(business, "incorporated_on")) or first_month)
    monthly_in = [0.0] * 6
    monthly_out = [0.0] * 6
    customers: Counter[str] = Counter()
    active_days: set[date] = set()
    balances: list[float] = []
    digital = cash = 0.0
    count = 0
    for txn in transactions:
        day = _day(_value(txn, "date"))
        if day is None or day >= as_of or day < first_month:
            continue
        idx = (day.year - first_month.year) * 12 + day.month - first_month.month
        if not 0 <= idx < 6:
            continue
        amount = float(_value(txn, "amount_pkr"))
        count += 1
        active_days.add(day)
        if _enum(_value(txn, "direction")) == "inflow":
            monthly_in[idx] += amount
            customers[str(_value(txn, "counterparty_hash"))] += amount
            if _enum(_value(txn, "channel")) == "cash_deposit":
                cash += amount
            else:
                digital += amount
        else:
            monthly_out[idx] += amount
        balance = _value(txn, "balance_after")
        if balance is not None:
            balances.append(float(balance))
    inflow = sum(monthly_in)
    avg_in = inflow / 6
    avg_net = (inflow - sum(monthly_out)) / 6
    installment = amount_requested_pkr / tenor_months
    balance_mean = (
        float(balance_override_pkr)
        if balance_override_pkr is not None
        else (sum(balances) / len(balances) if balances else 0.0)
    )
    balance_min = (
        float(balance_override_pkr)
        if balance_override_pkr is not None
        else (min(balances) if balances else 0.0)
    )
    if digital_share_override is None:
        digital_share = _divide(digital, inflow)
        cash_share = _divide(cash, inflow)
    else:
        digital_share = digital_share_override
        cash_share = 1 - digital_share_override
    bill_rows = []
    for bill in bills:
        due = _day(_value(bill, "due_date"))
        if due is not None and first_month <= due < as_of:
            paid = _day(_value(bill, "paid_date"))
            # A payment after the decision date is still unpaid at the decision.
            bill_rows.append((due, paid if paid is not None and paid < as_of else None))

    def available_bureau(record: Any) -> bool:
        stamp = _value(record, "created_at")
        if stamp is None:
            return True
        if isinstance(stamp, str):
            stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if bureau_cutoff is None:
            return stamp.date() < as_of
        return stamp.replace(tzinfo=None) <= bureau_cutoff.replace(tzinfo=None)

    bureau = [b for b in bureau_records if available_bureau(b)]
    bureau.sort(
        key=lambda b: (_day(_value(b, "created_at")) or date.min, int(_value(b, "id") or 0))
    )
    latest = bureau[-1] if bureau else None
    score = _value(latest, "score") if latest is not None else None
    incorporated = _day(_value(business, "incorporated_on")) or as_of
    vintage = max(
        0,
        (as_of.year - incorporated.year) * 12
        + as_of.month
        - incorporated.month
        - (as_of.day < incorporated.day),
    )
    mean_square = sum((x - avg_in) ** 2 for x in monthly_in) / 6
    trend = sum((i - 2.5) * x for i, x in enumerate(monthly_in)) / 17.5
    denominator = installment + (existing_debt_service_pkr or 0)
    observed_days = max(1, (as_of - max(first_month, start)).days)
    values = {
        "avg_monthly_inflow": avg_in,
        "inflow_cv": _divide(math.sqrt(mean_square), avg_in),
        "revenue_trend_slope": trend,
        "negative_cash_flow_months": float(
            sum(a < b for a, b in zip(monthly_in, monthly_out, strict=True))
        ),
        "avg_balance_ratio": _divide(balance_mean, avg_in),
        "min_balance_ratio": _divide(balance_min, avg_in),
        "dscr_proxy": _divide(avg_net, denominator) if denominator else 0.0,
        "instalment_to_inflow": _divide(installment, avg_in),
        "amount_to_annual_revenue": _divide(amount_requested_pkr, 12 * avg_in),
        "top3_customer_concentration": _divide(sum(v for _, v in customers.most_common(3)), inflow),
        "digital_payment_share": digital_share,
        "cash_deposit_share": cash_share,
        "active_day_ratio": len(active_days) / observed_days,
        "seasonality_index": _divide(max(monthly_in), avg_in),
        "on_time_bill_ratio": _divide(
            sum(p is not None and p <= d for d, p in bill_rows), len(bill_rows)
        ),
        "avg_days_late": _divide(
            sum(max(0, ((p or as_of) - d).days) for d, p in bill_rows), len(bill_rows)
        ),
        "vintage_months": float(vintage),
        "has_ntn": float(bool(_value(business, "has_ntn"))),
        "bureau_score": float(score) if score is not None else 600.0,
        "delinquency_flag": float(bool(_value(latest, "delinquent_flag"))) if latest else 0.0,
        "active_loan_count": float(_value(latest, "active_loans")) if latest else 0.0,
        "employees": float(_value(business, "employees")),
        "collateral_coverage": _divide(
            float(_value(business, "collateral_value_pkr")), amount_requested_pkr
        ),
        "has_bureau_record": float(latest is not None),
        "has_transaction_history": float(count > 0),
        "has_bill_history": float(bool(bill_rows)),
        "has_balance_data": float(bool(balances) or balance_override_pkr is not None),
        "has_debt_service_data": float(existing_debt_service_pkr is not None),
        "transaction_count": float(count),
        "avg_monthly_net_cash_flow": avg_net,
    }
    if set(values) != set(MODEL_FEATURES):
        raise ValueError("Feature implementation and registry differ")
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite feature value")
    return FeatureResult(application_id, FEATURE_VERSION, as_of, values)


def compute_features(application_id: int, session: Session | None = None) -> FeatureResult:
    """Compute and persist an immutable snapshot; caller-owned sessions are not committed."""
    owns_session = session is None
    if session is None:
        session = SessionLocal()
    try:
        app = session.get(Application, application_id)
        if app is None:
            raise ValueError(f"Application {application_id} does not exist")
        business = session.get(Business, app.business_id)
        as_of = _day(app.submitted_at or app.created_at)
        if as_of is None:
            raise ValueError("Application has no decision date")
        first_month = _month_shift(as_of, -5)
        txns = session.scalars(
            select(Transaction)
            .where(
                Transaction.business_id == app.business_id,
                Transaction.date >= first_month,
                Transaction.date < as_of,
            )
            .order_by(Transaction.date, Transaction.id)
        ).all()
        bills = session.scalars(
            select(BillPayment).where(
                BillPayment.business_id == app.business_id,
                BillPayment.due_date >= first_month,
                BillPayment.due_date < as_of,
            )
        ).all()
        bureau = session.scalars(
            select(BureauRecord).where(
                BureauRecord.business_id == app.business_id,
            )
        ).all()
        result = derive_features(
            application_id,
            app,
            business,
            txns,
            bills,
            bureau,
            as_of=as_of,
            bureau_cutoff=app.submitted_at or app.created_at,
        )
        snapshot = FeatureSnapshot(
            application_id=application_id, feature_version=FEATURE_VERSION, values=result.values
        )
        session.add(snapshot)
        session.flush()
        if owns_session:
            session.commit()
        return FeatureResult(application_id, FEATURE_VERSION, as_of, result.values, snapshot.id)
    except Exception:
        if owns_session:
            session.rollback()
        raise
    finally:
        if owns_session:
            session.close()
