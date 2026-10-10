"""Explainability reconciliation, bilingual coverage, and API persistence tests."""

import math

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Decision, ScoreResult
from app.db.seed import seed_demo
from app.db.session import init_db, make_engine
from app.enums import Outcome
from app.main import app
from app.ml.explain import (
    REASON_TEMPLATES,
    _render,
    adverse_action_notice,
    explain_values,
)
from app.ml.registry import MODEL_FEATURES
from app.ml.scorecard import load_model


def scenario_values():
    _, metadata = load_model()
    values = dict(zip(MODEL_FEATURES, metadata["training_reference"], strict=True))
    values.update(
        {
            "bureau_score": 300,
            "on_time_bill_ratio": 0.1,
            "avg_days_late": 20,
            "digital_payment_share": 0.1,
            "cash_deposit_share": 0.9,
            "instalment_to_inflow": 1.2,
        }
    )
    return values


def test_every_template_is_bilingual_utf8():
    assert set(REASON_TEMPLATES) == set(MODEL_FEATURES)
    for name, (english, urdu) in REASON_TEMPLATES.items():
        assert english and urdu
        assert urdu.encode("utf-8").decode("utf-8") == urdu
        for contribution in (-1.0, 1.0):
            en, ur = _render(name, 0.5, contribution)
            assert english in en
            assert urdu in ur
            assert ur.encode("utf-8").decode("utf-8") == ur


def test_logistic_waterfall_reconciles_and_notice_is_rule_based():
    explanation = explain_values(scenario_values(), check_stability=False)
    assert explanation.attribution_method == "standardized_coefficient"
    assert explanation.attribution_scale == "raw_log_odds"
    assert len(explanation.waterfall) == len(MODEL_FEATURES)
    assert math.isclose(
        explanation.baseline_log_odds + sum(r.contribution for r in explanation.waterfall),
        explanation.prediction_log_odds,
        abs_tol=1e-8,
    )
    assert len(explanation.top_risk_increasing) <= 5
    assert len(explanation.top_risk_reducing) <= 5
    assert len(explanation.global_importance) == len(MODEL_FEATURES)
    notice = adverse_action_notice(explanation, "rejected")
    assert notice.principal_features
    assert "declined" in notice.text_en
    assert "مسترد" in notice.text_ur
    assert "credit officer" in notice.text_en
    with pytest.raises(ValueError):
        adverse_action_notice(explanation, "approved")


def test_challenger_permutation_waterfall_reconciles():
    explanation = explain_values(
        scenario_values(),
        model_name="hist_gradient_boosting",
        check_stability=False,
    )
    assert explanation.attribution_method in {"permutation_shapley", "shap_tree"}
    assert math.isclose(
        explanation.baseline_log_odds + sum(r.contribution for r in explanation.waterfall),
        explanation.prediction_log_odds,
        abs_tol=1e-3,
    )


def test_tiny_noise_keeps_primary_reasons():
    explanation = explain_values(scenario_values())
    assert explanation.reasons_stable
    assert explanation.stability_agreement == 1.0


def test_endpoint_persists_reasons(monkeypatch):
    engine = make_engine("sqlite://")
    init_db(engine, seed_hook=seed_demo)
    with Session(engine) as session:
        session.add(
            Decision(
                application_id=1,
                outcome=Outcome.REFER,
                amount=100000,
                tenor=12,
                rate_band="demo",
                decided_by=1,
                policy_version="demo-v1",
            )
        )
        session.commit()
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr("app.ml.explain.SessionLocal", factory)
    response = TestClient(app).get("/api/v1/applications/1/explanation")
    assert response.status_code == 200
    data = response.json()
    assert data["score_result_id"] is not None
    assert len(data["waterfall"]) == len(MODEL_FEATURES)
    assert data["top_risk_increasing"][0]["text_ur"]
    assert data["adverse_action_notice"]["outcome"] == "referred"
    assert data["adverse_action_notice"]["principal_features"]
    with Session(engine) as session:
        record = session.scalar(select(ScoreResult))
        assert record.id == data["score_result_id"]
        assert record.feature_snapshot_id
        assert record.reasons
        assert record.reasons[0]["text_en"]
    engine.dispose()


def test_three_synthetic_personas_have_readable_reasons():
    from app.ml.features import compute_features

    ayesha = explain_values(compute_features(1).values)
    karim = explain_values(compute_features(2).values)
    fastfix = explain_values(compute_features(3).values)
    assert "on_time_bill_ratio" in {item.feature for item in ayesha.top_risk_reducing}
    assert "digital_payment_share" in {item.feature for item in ayesha.top_risk_reducing}
    assert {"seasonality_index", "avg_days_late"} <= set(
        adverse_action_notice(karim, "referred").principal_features
    )
    assert {"on_time_bill_ratio", "bureau_score"} <= {
        item.feature for item in fastfix.top_risk_increasing
    }
    assert all(item.reasons_stable for item in (ayesha, karim, fastfix))
