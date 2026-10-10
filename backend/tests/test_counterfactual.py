"""Counterfactual plans must remain actionable and pass the actual decision engine."""

from time import perf_counter

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.v1.counterfactuals import PathResponse
from app.db.models import Application, FeatureSnapshot, ScoreResult
from app.db.session import SessionLocal
from app.main import app
from app.ml.counterfactual import (
    ACTION_FEATURE,
    ACTION_TEXT,
    ScenarioChanges,
    find_path_to_approval,
    project_features,
    what_if,
)
from app.ml.registry import FEATURES
from app.services.pipeline import warm_models


def test_karim_gets_three_verified_bilingual_plans():
    warm_models()
    start = perf_counter()
    result = find_path_to_approval(2, lender_view=True)
    assert perf_counter() - start < 2
    assert result["original_decision"] == "REFER"
    assert [plan["label"] for plan in result["plans"]] == ["Fastest", "Cheapest", "Strongest"]
    assert any(plan["changes"].get("amount_requested_pkr") == 1500000 for plan in result["plans"])
    for plan in result["plans"]:
        assert plan["lender_view"]["policy_accepted"]
        assert all(FEATURES[ACTION_FEATURE[key]].actionable for key in plan["changes"])
        assert all(step["text_en"] and step["text_ur"].encode("utf-8") for step in plan["steps"])
        preview = what_if(2, ScenarioChanges(**plan["changes"]))
        assert preview["what_if"]["decision"] == "APPROVE"
        assert preview["what_if"]["pd"] <= 0.095
        assert preview["what_if"]["recommended_amount_pkr"] == plan["expected_approved_amount_pkr"]
    PathResponse.model_validate(result)
    with SessionLocal() as session:
        row = session.scalar(
            select(ScoreResult)
            .where(ScoreResult.application_id == 2)
            .order_by(ScoreResult.id.desc())
        )
        assert len(row.counterfactuals) == 3


def test_what_if_is_fast_and_rejects_forbidden_edits():
    warm_models()
    client = TestClient(app)
    start = perf_counter()
    response = client.post("/api/v1/applications/2/what-if", json={"amount_requested_pkr": 1500000})
    assert perf_counter() - start < 0.5
    assert response.status_code == 200
    assert response.json()["what_if"]["decision"] == "APPROVE"
    assert (
        client.post("/api/v1/applications/2/what-if", json={"vintage_months": 120}).status_code
        == 422
    )
    assert (
        client.post("/api/v1/applications/2/what-if", json={"digital_payment_share": 2}).status_code
        == 422
    )


def test_scenario_dependencies_and_immutable_facts():
    with SessionLocal() as session:
        app_row = session.get(Application, 2)
        score = session.scalar(
            select(ScoreResult)
            .where(ScoreResult.application_id == 2)
            .order_by(ScoreResult.id.desc())
        )
        base = session.get(FeatureSnapshot, score.feature_snapshot_id).values
        changes = ScenarioChanges(
            amount_requested_pkr=1500000, tenor_months=24, digital_payment_share=0.8
        )
        projected = project_features(base, app_row, changes)
        assert projected["instalment_to_inflow"] == 1500000 / 24 / base["avg_monthly_inflow"]
        assert projected["dscr_proxy"] == base["avg_monthly_net_cash_flow"] / (1500000 / 24)
        assert projected["amount_to_annual_revenue"] == 1500000 / (12 * base["avg_monthly_inflow"])
        assert (
            projected["cash_deposit_share"] == 0.2
            or abs(projected["cash_deposit_share"] - 0.2) < 1e-12
        )
        for name, spec in FEATURES.items():
            if not spec.actionable:
                assert projected[name] == base[name]


def test_every_action_has_renderable_bilingual_text():
    assert set(ACTION_TEXT) == set(ACTION_FEATURE)
    for english, urdu in ACTION_TEXT.values():
        assert english.format(from_=100, to=200)
        assert urdu.format(from_=100, to=200).encode("utf-8")
