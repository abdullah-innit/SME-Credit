"""Application scoring and system recommendation pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import enums
from app.db.models import Application, AuditLog, Business, Decision, Document, ScoreResult, User
from app.db.session import SessionLocal
from app.ml.explain import explain_values
from app.ml.features import compute_features
from app.ml.scorecard import ARTIFACTS, calibrated_probability, load_model, pd_to_score, tier_for_pd
from app.services.policy import evaluate_policy, load_policy


@lru_cache(maxsize=1)
def _models() -> tuple[dict, dict, dict]:
    champion, metadata = load_model()
    challenger = joblib.load(ARTIFACTS / "hist_gradient_boosting.joblib")
    return champion, challenger, metadata


def warm_models() -> None:
    """Load artifacts and initialize the challenger before serving requests."""
    _, challenger, metadata = _models()
    vector = np.zeros((1, len(metadata["feature_list"])), dtype=float)
    calibrated_probability(challenger, vector)


def integrity_hook(application_id: int) -> dict:
    """Prompt 13 replaces this with evidence-backed integrity screening."""
    return {"status": "pending", "flags": [], "application_id": application_id}


def _documents(session: Session, business_id: int) -> list[Document]:
    return session.scalars(
        select(Document).where(Document.business_id == business_id).order_by(Document.id)
    ).all()


def _evidence_fingerprint(app: Application, documents: list[Document]) -> str:
    payload = {
        "amount": app.amount_requested_pkr,
        "tenor": app.tenor_months,
        "submitted_at": str(app.submitted_at),
        "documents": [
            (
                doc.id,
                doc.sha256,
                doc.parse_status.value,
                (doc.parse_report or {}).get("data_quality_score"),
            )
            for doc in documents
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _data_quality(values: dict[str, float], documents: list[Document]) -> dict:
    present = [
        values[key]
        for key in (
            "has_transaction_history",
            "has_bill_history",
            "has_balance_data",
            "has_bureau_record",
            "has_debt_service_data",
        )
    ]
    scores = [
        float(doc.parse_report["data_quality_score"])
        for doc in documents
        if doc.parse_status == enums.ParseStatus.PARSED
        and isinstance((doc.parse_report or {}).get("data_quality_score"), (int, float))
    ]
    statement_score = min(scores) if scores else None
    base = sum(present) / len(present)
    completeness = base if statement_score is None else 0.7 * base + 0.3 * statement_score / 100
    return {
        "completeness": round(completeness, 2),
        "transaction_count": int(values["transaction_count"]),
        "known_debt_service": bool(values["has_debt_service_data"]),
        "statement_quality_score": statement_score,
        "statement_count": len(scores),
    }


def score_application(
    application_id: int, *, session: Session | None = None, policy_path: Path | None = None
) -> dict:
    own = session is None
    if session is None:
        session = SessionLocal()
    try:
        app = session.get(Application, application_id)
        if app is None:
            raise LookupError("Application not found")
        policy = load_policy(policy_path) if policy_path else load_policy()
        champion, challenger, metadata = _models()
        model_version = metadata["model_version"]
        policy_version = policy["effective_version"]
        documents = _documents(session, app.business_id)
        evidence_fingerprint = _evidence_fingerprint(app, documents)
        for log in session.scalars(
            select(AuditLog)
            .where(
                AuditLog.action == "system_recommendation",
                AuditLog.entity == "applications",
                AuditLog.entity_id == application_id,
            )
            .order_by(AuditLog.id.desc())
        ):
            stored = log.after or {}
            if (
                stored.get("model_version") == model_version
                and stored.get("policy_version") == policy_version
                and stored.get("evidence_fingerprint") == evidence_fingerprint
            ):
                return {**stored["response"], "idempotent_replay": True}
        snapshot = compute_features(application_id, session)
        values = snapshot.values
        vector = np.array([[values[name] for name in metadata["feature_list"]]], dtype=float)
        pd = float(calibrated_probability(champion, vector)[0])
        challenger_pd = float(calibrated_probability(challenger, vector)[0])
        score = pd_to_score(pd, metadata["scorecard"])
        tier = tier_for_pd(pd, metadata["scorecard"])
        business = session.get(Business, app.business_id)
        result = evaluate_policy(
            policy=policy,
            values=values,
            sector=business.sector,
            requested_amount_pkr=app.amount_requested_pkr,
            requested_tenor_months=app.tenor_months,
            tier=tier,
            champion_pd=pd,
            challenger_pd=challenger_pd,
        )
        explanation = explain_values(
            values,
            application_id=application_id,
            loaded=(champion, metadata),
            check_stability=False,
        )
        stored_tier = (
            enums.RiskTier.LOW
            if tier in {"A", "B"}
            else enums.RiskTier.MEDIUM
            if tier == "C"
            else enums.RiskTier.HIGH
        )
        reasons = [
            asdict(item)
            for item in (*explanation.top_risk_increasing, *explanation.top_risk_reducing)
        ]
        score_row = ScoreResult(
            application_id=application_id,
            feature_snapshot_id=snapshot.snapshot_id,
            model_version=model_version,
            pd=pd,
            score=score,
            tier=stored_tier,
            reasons=reasons,
            counterfactuals=[],
        )
        session.add(score_row)
        session.flush()
        # A dedicated system actor satisfies the existing non-null decision FK.
        actor_log = session.scalar(
            select(AuditLog).where(AuditLog.action == "system_actor_created").limit(1)
        )
        if actor_log is None:
            actor = User(role=enums.UserRole.ADMIN)
            session.add(actor)
            session.flush()
            session.add(
                AuditLog(
                    actor=actor.id,
                    action="system_actor_created",
                    entity="users",
                    entity_id=actor.id,
                    before=None,
                    after={"role": "system_recommendation_actor"},
                    at=datetime.now(UTC),
                )
            )
        else:
            actor = session.get(User, actor_log.entity_id)
        mapped = {
            "APPROVE": enums.Outcome.APPROVE,
            "REFER": enums.Outcome.REFER,
            "DECLINE": enums.Outcome.REJECT,
        }
        decision = Decision(
            application_id=application_id,
            outcome=mapped[result.outcome],
            amount=result.recommended_amount_pkr,
            tenor=result.tenor_months,
            rate_band=result.indicative_rate_band,
            decided_by=actor.id,
            policy_version=policy_version,
            is_override=False,
        )
        session.add(decision)
        session.flush()
        checks = [asdict(c) for c in result.checks]
        failed = [c["code"] for c in checks if c["status"] == "fail"]
        assurance = {
            "calibration_method": champion["method"],
            "test_brier_score": metadata["metrics"]["models"][metadata["champion"]]["test"]["all"][
                "brier"
            ],
            "confidence": "limited: synthetic-data calibration",
            "model_agreement": abs(pd - challenger_pd)
            <= policy["maximum_champion_challenger_pd_gap"],
            "pd_gap": round(abs(pd - challenger_pd), 4),
            "data_quality": _data_quality(values, documents),
            "policy_summary": {
                state: sum(c["status"] == state for c in checks)
                for state in ("pass", "fail", "warn")
            },
        }
        response = {
            "application_id": application_id,
            "score_result_id": score_row.id,
            "decision_id": decision.id,
            "model_version": model_version,
            "policy_version": policy_version,
            "recommendation_only": True,
            "idempotent_replay": False,
            "champion_pd": pd,
            "challenger_pd": challenger_pd,
            "score": score,
            "tier": tier,
            "outcome": result.outcome,
            "reason_codes": failed,
            "policy_checks": checks,
            "offer": {
                "recommended_amount_pkr": result.recommended_amount_pkr,
                "tenor_months": result.tenor_months,
                "indicative_rate_band": result.indicative_rate_band,
                "expected_loss_pkr": result.expected_loss_pkr,
            },
            "integrity": integrity_hook(application_id),
            "assurance": assurance,
        }
        session.add(
            AuditLog(
                actor=actor.id,
                action="system_recommendation",
                entity="applications",
                entity_id=application_id,
                before=None,
                after={
                    "model_version": model_version,
                    "policy_version": policy_version,
                    "evidence_fingerprint": evidence_fingerprint,
                    "response": response,
                },
                at=datetime.now(UTC),
            )
        )
        if own:
            session.commit()
        return response
    except Exception:
        if own:
            session.rollback()
        raise
    finally:
        if own:
            session.close()
