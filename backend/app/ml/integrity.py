"""Fraud and integrity engine for SME Credit."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import enums
from app.db.models import (
    Application,
    Business,
    Document,
    FraudFlag,
    IntegrityResult,
    Transaction,
)
from app.db.session import SessionLocal
from app.ml.scorecard import ARTIFACTS

INTEGRITY_MODEL_PATH = ARTIFACTS / "isolation_forest.joblib"


def get_benford_expected() -> dict[int, float]:
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def compute_benford_mad(amounts: Iterable[float]) -> float:
    counts = Counter()
    total = 0
    for amount in amounts:
        if amount <= 0:
            continue
        first_digit = int(str(int(amount)).lstrip("0")[0])
        counts[first_digit] += 1
        total += 1
    if total == 0:
        return 0.0
    expected = get_benford_expected()
    mad = 0.0
    for d in range(1, 10):
        obs = counts[d] / total
        mad += abs(obs - expected[d])
    return mad / 9


def compute_window_features(transactions: list[Transaction], as_of: date) -> list[float]:
    """Compute features for the IsolationForest anomaly model."""
    if not transactions:
        return [0.0] * 7
    inflows = [t.amount_pkr for t in transactions if t.direction.value == "inflow"]
    outflows = [t.amount_pkr for t in transactions if t.direction.value == "outflow"]
    total_in = sum(inflows)
    total_out = sum(outflows)
    count = len(transactions)
    weekend_count = sum(1 for t in transactions if t.date.weekday() >= 5)
    mean_in = total_in / len(inflows) if inflows else 0.0
    mean_out = total_out / len(outflows) if outflows else 0.0
    return [
        float(total_in),
        float(total_out),
        float(count),
        float(weekend_count / count),
        float(mean_in),
        float(mean_out),
        float(len(inflows) / count if count else 0.0),
    ]


def evaluate_rules(
    application_id: int,
    business_id: int,
    transactions: list[Transaction],
    as_of: date,
    session: Session,
    amount_requested_pkr: int,
) -> list[dict[str, Any]]:
    """Evaluate business logic rules for fraud."""
    flags = []

    # Sort transactions by date and id
    sorted_txns = sorted(transactions, key=lambda t: (t.date, t.id))

    # 1. Balance reconciliation breaks
    reconciliation_breaks = []
    for i, t in enumerate(sorted_txns):
        if t.balance_after is None:
            continue
        if i > 0 and sorted_txns[i - 1].balance_after is not None:
            prev_balance = sorted_txns[i - 1].balance_after
            delta = t.amount_pkr if t.direction.value == "inflow" else -t.amount_pkr
            expected = prev_balance + delta
            if expected != t.balance_after:
                reconciliation_breaks.append(t.id)
    if reconciliation_breaks:
        flags.append(
            {
                "severity": enums.Severity.HIGH,
                "code": "RECONCILIATION_BREAK",
                "evidence": {"transaction_ids": reconciliation_breaks[:10]},
            }
        )

    # 2. Window dressing (spike 30 days before app with withdrawals)
    window_30d = as_of - timedelta(days=30)
    window_txns = [t for t in sorted_txns if window_30d <= t.date < as_of]
    inflows_30d = [t.amount_pkr for t in window_txns if t.direction.value == "inflow"]
    outflows_30d = [t.amount_pkr for t in window_txns if t.direction.value == "outflow"]
    prior_txns = [t for t in sorted_txns if t.date < window_30d]
    prior_inflows = [t.amount_pkr for t in prior_txns if t.direction.value == "inflow"]
    avg_prior_inflow = sum(prior_inflows) / (len(prior_inflows) or 1)

    if inflows_30d and avg_prior_inflow > 0:
        spike_inflows = [
            t
            for t in window_txns
            if t.direction.value == "inflow" and t.amount_pkr > 3 * avg_prior_inflow
        ]
        if spike_inflows and sum(outflows_30d) > sum(inflows_30d) * 0.8:
            flags.append(
                {
                    "severity": enums.Severity.HIGH,
                    "code": "WINDOW_DRESSING",
                    "evidence": {"spike_transaction_ids": [t.id for t in spike_inflows][:10]},
                }
            )

    # 3. Circular flows (same counterparty in and out within 48 hours)
    circular = []
    for i, t_in in enumerate(sorted_txns):
        if t_in.direction.value == "inflow":
            for t_out in sorted_txns[i + 1 :]:
                if (t_out.date - t_in.date).days > 2:
                    break
                if (
                    t_out.direction.value == "outflow"
                    and t_out.counterparty_hash == t_in.counterparty_hash
                    and abs(t_out.amount_pkr - t_in.amount_pkr) / float(t_in.amount_pkr) < 0.05
                ):
                    circular.append((t_in.id, t_out.id))
    if circular:
        flags.append(
            {
                "severity": enums.Severity.HIGH,
                "code": "CIRCULAR_FLOWS",
                "evidence": {"transaction_pairs": circular[:5]},
            }
        )

    # 4. Round-number deposit share
    inflow_txns = [t for t in sorted_txns if t.direction.value == "inflow"]
    if inflow_txns:
        # Check divisibility by 5000 or 10000
        round_count = sum(
            1 for t in inflow_txns if t.amount_pkr >= 5000 and t.amount_pkr % 5000 == 0
        )
        round_share = round_count / len(inflow_txns)
        if round_share > 0.4:
            flags.append(
                {
                    "severity": enums.Severity.MEDIUM,
                    "code": "ROUND_NUMBER_SHARE",
                    "evidence": {"round_share": round(round_share, 3)},
                }
            )

    # 5. Benford first-digit deviation
    if len(inflow_txns) > 20:
        mad = compute_benford_mad([t.amount_pkr for t in inflow_txns])
        if mad > 0.06:
            flags.append(
                {
                    "severity": enums.Severity.MEDIUM,
                    "code": "BENFORD_DEVIATION",
                    "evidence": {"mad": round(mad, 4)},
                }
            )

    # 6. Duplicate or impossible timestamps
    impossible = [t.id for t in sorted_txns if t.date > date.today()]
    date_amounts = Counter((t.date, t.amount_pkr, t.direction.value) for t in sorted_txns)
    duplicates = [items for items, count in date_amounts.items() if count >= 10]
    if impossible or duplicates:
        flags.append(
            {
                "severity": enums.Severity.MEDIUM,
                "code": "TIMESTAMP_ANOMALY",
                "evidence": {
                    "impossible_ids": impossible[:5],
                    "duplicate_patterns": len(duplicates),
                },
            }
        )

    # 7. Abnormal night and weekend activity
    weekend_txns = [t for t in sorted_txns if t.date.weekday() >= 5]
    if len(sorted_txns) > 0 and len(weekend_txns) / len(sorted_txns) > 0.5:
        flags.append(
            {
                "severity": enums.Severity.LOW,
                "code": "WEEKEND_ACTIVITY",
                "evidence": {"weekend_share": round(len(weekend_txns) / len(sorted_txns), 3)},
            }
        )

    # 8. Declared versus observed revenue mismatch
    total_inflows = sum(t.amount_pkr for t in inflow_txns)
    months = max(1, (as_of - sorted_txns[0].date).days / 30.0) if sorted_txns else 1
    avg_monthly = total_inflows / months
    if avg_monthly > 0 and amount_requested_pkr > avg_monthly * 12 * 5:
        flags.append(
            {
                "severity": enums.Severity.MEDIUM,
                "code": "REVENUE_MISMATCH",
                "evidence": {
                    "avg_monthly": round(avg_monthly, 2),
                    "requested": amount_requested_pkr,
                },
            }
        )

    # 9. Shell-cluster signal
    app = session.get(Application, application_id)
    if app and app.business:
        docs = session.scalars(select(Document).where(Document.business_id == business_id)).all()
        doc_hashes = [d.sha256 for d in docs]
        if doc_hashes:
            other_docs = session.scalars(
                select(Document).where(
                    Document.sha256.in_(doc_hashes), Document.business_id != business_id
                )
            ).all()
            if other_docs:
                shared = list({d.sha256 for d in other_docs})[:5]
                flags.append(
                    {
                        "severity": enums.Severity.HIGH,
                        "code": "SHELL_CLUSTER",
                        "evidence": {"shared_document_hashes": shared},
                    }
                )

    return flags


def train_anomaly_model(session: Session) -> None:
    from sklearn.ensemble import IsolationForest

    businesses = session.scalars(select(Business).where(Business.is_synthetic)).all()
    features_list = []
    for b in businesses:
        txns = session.scalars(select(Transaction).where(Transaction.business_id == b.id)).all()
        app = session.scalars(select(Application).where(Application.business_id == b.id)).first()
        if not app:
            continue
        as_of = app.submitted_at.date() if app.submitted_at else date.today()
        feats = compute_window_features(list(txns), as_of)
        features_list.append(feats)

    if features_list:
        X = np.array(features_list)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        clf.fit(X)

        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, INTEGRITY_MODEL_PATH)


def load_anomaly_model() -> Any:
    if not INTEGRITY_MODEL_PATH.exists():
        return None
    return joblib.load(INTEGRITY_MODEL_PATH)


def evaluate_integrity(application_id: int, *, session: Session | None = None) -> dict[str, Any]:
    own = session is None
    if session is None:
        session = SessionLocal()

    try:
        app = session.get(Application, application_id)
        if app is None:
            raise LookupError("Application not found")

        txns = session.scalars(
            select(Transaction).where(Transaction.business_id == app.business_id)
        ).all()
        as_of = app.submitted_at.date() if app.submitted_at else date.today()

        flags_data = evaluate_rules(
            application_id, app.business_id, list(txns), as_of, session, app.amount_requested_pkr
        )

        model = load_anomaly_model()
        anomaly_score = 0
        if model:
            feats = compute_window_features(list(txns), as_of)
            X = np.nan_to_num(np.array([feats]), nan=0.0, posinf=0.0, neginf=0.0)
            raw_score = model.decision_function(X)[0]
            norm = (raw_score + 0.3) / 0.6
            anomaly_score = int(np.clip(norm * 100, 0, 100))
        else:
            anomaly_score = 80

        penalty = 0
        for f in flags_data:
            if f["severity"] == enums.Severity.HIGH:
                penalty += 40
            elif f["severity"] == enums.Severity.MEDIUM:
                penalty += 15
            elif f["severity"] == enums.Severity.LOW:
                penalty += 5

        final_score = max(0, anomaly_score - penalty)

        highest_severity = enums.Severity.LOW
        if flags_data:
            for sev in [enums.Severity.HIGH, enums.Severity.MEDIUM, enums.Severity.LOW]:
                if any(f["severity"] == sev for f in flags_data):
                    highest_severity = sev
                    break
        else:
            if final_score < 40:
                highest_severity = enums.Severity.MEDIUM
                flags_data.append(
                    {
                        "severity": enums.Severity.MEDIUM,
                        "code": "ANOMALY_MODEL",
                        "evidence": {"anomaly_score": anomaly_score},
                    }
                )

        result = IntegrityResult(
            application_id=application_id,
            severity=highest_severity if flags_data else enums.Severity.LOW,
            code="INTEGRITY_COMPLETED",
            evidence={"final_score": final_score, "anomaly_score": anomaly_score},
        )
        session.add(result)
        session.flush()

        for f in flags_data:
            flag = FraudFlag(
                application_id=application_id,
                integrity_result_id=result.id,
                severity=f["severity"],
                code=f["code"],
                evidence=f["evidence"],
            )
            session.add(flag)

        if own:
            session.commit()

        return {
            "application_id": application_id,
            "status": "completed",
            "integrity_score": final_score,
            "flags": flags_data,
        }
    except Exception:
        if own:
            session.rollback()
        raise
    finally:
        if own:
            session.close()
