"""Decision pipeline contract and policy regression tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi.testclient import TestClient

from app.db.models import Application
from app.db.session import SessionLocal
from app.main import app
from app.services.pipeline import score_application, warm_models
from app.services.policy import evaluate_policy, load_policy


def test_persona_decisions_and_idempotency():
    for application_id, expected in ((1, "APPROVE"), (2, "REFER"), (3, "REFER")):
        first = score_application(application_id)
        replay = score_application(application_id)
        assert first["outcome"] == expected
        assert replay["idempotent_replay"]
        assert replay["score_result_id"] == first["score_result_id"]
        assert replay["decision_id"] == first["decision_id"]
        assert first["recommendation_only"]
        assert all(check["code"] for check in first["policy_checks"] if check["status"] == "fail")
    assert "TIER_EXPOSURE" in score_application(2)["reason_codes"]


def test_score_endpoint_and_warm_latency():
    client = TestClient(app)
    start = perf_counter()
    response = client.post("/api/v1/applications/1/score")
    assert perf_counter() - start < 1.0
    assert response.status_code == 200
    assert response.json()["outcome"] == "APPROVE"
    assert response.json()["assurance"]["confidence"].startswith("limited")


def test_new_application_scores_under_one_second_after_startup():
    warm_models()
    with SessionLocal() as session:
        source = session.get(Application, 1)
        fresh = Application(
            business_id=source.business_id,
            product=source.product,
            amount_requested_pkr=source.amount_requested_pkr,
            tenor_months=source.tenor_months,
            purpose=source.purpose,
            status=source.status,
            submitted_at=source.submitted_at,
            consent_id=source.consent_id,
        )
        session.add(fresh)
        session.flush()
        start = perf_counter()
        result = score_application(fresh.id, session=session)
        assert perf_counter() - start < 1.0
        assert result["outcome"] == "APPROVE"
        session.rollback()


def test_policy_edit_changes_version_and_outcome():
    source = Path(__file__).resolve().parents[1] / "app" / "config" / "policy.yaml"
    with TemporaryDirectory() as directory:
        path = Path(directory) / "policy.yaml"
        original = source.read_text(encoding="utf-8")
        path.write_text(original, encoding="utf-8")
        old = load_policy(path)
        path.write_text(
            original.replace("minimum_vintage_months: 6", "minimum_vintage_months: 100"),
            encoding="utf-8",
        )
        edited = load_policy(path)
        assert old["effective_version"] != edited["effective_version"]
        values = {
            "avg_monthly_net_cash_flow": 150000.0,
            "avg_monthly_inflow": 500000.0,
            "vintage_months": 42.0,
            "dscr_proxy": 2.0,
            "instalment_to_inflow": 0.1,
            "has_debt_service_data": 1.0,
        }
        common = dict(
            values=values,
            sector="garments",
            requested_amount_pkr=500000,
            requested_tenor_months=12,
            tier="A",
            champion_pd=0.02,
            challenger_pd=0.03,
        )
        assert evaluate_policy(policy=old, **common).outcome == "APPROVE"
        denied = evaluate_policy(policy=edited, **common)
        assert denied.outcome == "DECLINE"
        assert "MIN_VINTAGE" in [c.code for c in denied.checks if c.status == "fail"]
