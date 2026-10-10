"""Acceptance tests for Prompt 11: AI credit memo and plain-language explanations."""

import json
import time

from fastapi.testclient import TestClient

from app.db.models import MemoRecord
from app.db.session import SessionLocal
from app.llm.grounding import check_numbers
from app.main import app
from app.services.memo import (
    MEMO_LABEL_AI,
    MEMO_LABEL_HUMAN,
    build_facts_bundle,
    get_or_create_memo,
)

client = TestClient(app)


def test_memo_generation_personas_latency_and_sections():
    """Verify memo generation for all three demo personas under 10 seconds (instantly offline)."""
    persona_apps = [1, 2, 3]  # Ayesha, Karim, FastFix

    for app_id in persona_apps:
        start_time = time.time()
        res = client.get(f"/api/v1/applications/{app_id}/memo?force_regenerate=true")
        elapsed = time.time() - start_time

        assert res.status_code == 200, f"Failed for app_id {app_id}: {res.text}"
        assert elapsed < 10.0, f"Memo generation for persona {app_id} took {elapsed:.2f}s (max 10s)"

        data = res.json()
        assert "memo_data" in data
        memo = data["memo_data"]

        # Required label
        assert memo["label"] == MEMO_LABEL_AI

        # Check all 8 bank-facing sections
        assert "executive_summary_en" in memo and len(memo["executive_summary_en"]) > 30
        assert "executive_summary_ur" in memo and len(memo["executive_summary_ur"]) > 30
        assert "business_snapshot" in memo
        assert "cash_flow_analysis" in memo
        assert "key_risk_drivers" in memo
        assert "integrity_findings" in memo
        assert "policy_checks" in memo
        assert "recommendation_and_conditions" in memo
        assert "applicant_questions" in memo
        assert len(memo["applicant_questions"]) >= 3
        assert "footnotes" in memo and len(memo["footnotes"]) >= 3


def test_zero_numbers_outside_facts_bundle():
    """Verify number grounding checker passes with zero ungrounded numbers."""
    with SessionLocal() as session:
        for app_id in [1, 2, 3]:
            facts = build_facts_bundle(app_id, session)
            memo_dict = get_or_create_memo(app_id, session, force_regenerate=True)
            memo_data = memo_dict["memo_data"]

            # Grounding check on serialized memo data against facts
            violations = check_numbers(json.dumps(memo_data, ensure_ascii=False), facts)
            assert violations == [], f"Found ungrounded numbers in app {app_id}: {violations}"


def test_applicant_explanation_plain_language_and_bilingual():
    """Applicant explanation is plain-language, bilingual (Grade 6-8), and has Roman Urdu toggle."""
    for app_id in [1, 2, 3]:
        res = client.get(f"/api/v1/applications/{app_id}/explanation/applicant")
        assert res.status_code == 200
        exp = res.json()

        assert "plain_english" in exp and len(exp["plain_english"]) > 20
        assert "plain_urdu" in exp and len(exp["plain_urdu"]) > 20
        assert "plain_roman_urdu" in exp and len(exp["plain_roman_urdu"]) > 20
        assert exp["reading_level"] == "Grade 6-8 Plain Language"

        # Adverse action reasons and Path to Approval for non-approved
        if exp["status"] in {"REFER", "DECLINE"}:
            assert "adverse_action_reasons_en" in exp
            assert len(exp["adverse_action_reasons_en"]) >= 1
            assert "adverse_action_reasons_ur" in exp
            assert "adverse_action_reasons_roman" in exp


def test_officer_editing_versioning_and_diff():
    """Officers can edit memo; versions are stored in MemoRecord with label and diff."""
    app_id = 1
    # Clear any previous records for app 1 to ensure clean v1 -> v2 transition
    with SessionLocal() as s:
        s.query(MemoRecord).filter(MemoRecord.application_id == app_id).delete()
        s.commit()

    # 1. Initial generation (v1)
    get_res = client.get(f"/api/v1/applications/{app_id}/memo?force_regenerate=true")
    assert get_res.status_code == 200
    v1_data = get_res.json()
    assert v1_data["memo_data"]["label"] == MEMO_LABEL_AI

    # 2. Officer edits executive summary
    edited_data = dict(v1_data["memo_data"])
    edited_data["executive_summary_en"] = (
        "Officer amended summary: Approved subject to bank statement refresh."
    )
    put_res = client.put(
        f"/api/v1/applications/{app_id}/memo",
        json={"memo_data": edited_data, "user_id": 2},
    )
    assert put_res.status_code == 200
    v2_data = put_res.json()

    assert v2_data["label"] == MEMO_LABEL_HUMAN
    assert v2_data["version"] == "v2"
    assert "Officer amended summary" in v2_data["memo_data"]["executive_summary_en"]
    assert "diff" in v2_data and len(v2_data["diff"]) > 0
    assert "memo_v1.txt" in v2_data["diff"]

    # 3. List versions
    ver_res = client.get(f"/api/v1/applications/{app_id}/memo/versions")
    assert ver_res.status_code == 200
    versions = ver_res.json()
    assert len(versions) >= 2
    assert versions[0]["label"] == MEMO_LABEL_HUMAN  # most recent first


def test_regeneration_tone_options():
    """Regeneration on demand supports concise and detailed tone options."""
    app_id = 1

    # Concise tone
    res_concise = client.post(
        f"/api/v1/applications/{app_id}/memo/generate",
        json={"tone": "concise", "user_id": 1},
    )
    assert res_concise.status_code == 200
    concise_data = res_concise.json()
    assert concise_data["tone"] == "concise"

    # Detailed tone
    res_detailed = client.post(
        f"/api/v1/applications/{app_id}/memo/generate",
        json={"tone": "detailed", "user_id": 1},
    )
    assert res_detailed.status_code == 200
    detailed_data = res_detailed.json()
    assert detailed_data["tone"] == "detailed"


def test_export_html_and_pdf():
    """Export endpoints return print-optimised HTML with RTL Urdu and valid PDF bytes."""
    app_id = 1

    # 1. HTML export
    html_res = client.get(f"/api/v1/applications/{app_id}/memo/export/html")
    assert html_res.status_code == 200
    assert "text/html" in html_res.headers["content-type"]
    html_text = html_res.text
    assert MEMO_LABEL_AI in html_text
    assert 'class="urdu"' in html_text
    assert "direction: rtl" in html_text
    assert "@media print" in html_text

    # 2. PDF export
    pdf_res = client.get(f"/api/v1/applications/{app_id}/memo/export/pdf")
    assert pdf_res.status_code == 200
    assert "application/pdf" in pdf_res.headers["content-type"]
    assert pdf_res.content.startswith(b"%PDF-")
