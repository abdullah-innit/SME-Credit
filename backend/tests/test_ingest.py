"""Statement parser, reconciliation, privacy and API regression tests."""

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import Workbook

from app.main import app
from app.services.ingest.normalise import normalise_rows
from app.services.ingest.parsers import map_columns, parse_rows
from app.services.ingest.service import ingest_statement
from app.services.ingest.validate import validate_statement
from app.services.pipeline import score_application

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def test_all_six_samples_parse_and_reconcile():
    for path in sorted(SAMPLES.glob("*_statement.*")):
        if path.suffix not in {".csv", ".pdf"}:
            continue
        rows, mapping, adapter = parse_rows(path.read_bytes(), path.suffix)
        normalised, errors = normalise_rows(rows, mapping, business_id=1)
        result = validate_statement(normalised, errors)
        assert adapter
        assert len(result.unique_rows) == (730 if path.suffix == ".csv" else 45)
        assert result.report["balance_reconciliation"]["passed"]
        assert result.report["data_quality_score"] == (90 if "fastfix" in path.name else 100)
        assert all(
            row.category_source in {"rule", "llm"} and 0 <= row.category_confidence <= 1
            for row in normalised
        )


def test_xlsx_debit_credit_adapter():
    book = Workbook()
    sheet = book.active
    sheet.append(["Date", "Narration", "Debit", "Credit", "Balance"])
    sheet.append(["2026-09-01", "Customer sale", 0, 500, 1500])
    sheet.append(["2026-09-02", "Supplier payment", 200, 0, 1300])
    stream = BytesIO()
    book.save(stream)
    rows, mapping, adapter = parse_rows(stream.getvalue(), ".xlsx")
    result = validate_statement(*normalise_rows(rows, mapping, business_id=1))
    assert adapter == "debit_credit"
    assert [row.category for row in result.unique_rows] == ["sales", "supplier_payment"]
    assert result.report["balance_reconciliation"]["passed"]


def test_signed_amount_generic_mapper_and_invalid_llm_fallback(monkeypatch):
    raw = (
        b"Transaction Date,Particulars,Transaction Amount,Running Balance\n"
        b"2026-09-01,Sales,500,1500\n"
        b"2026-09-02,Rent,-200,1300\n"
    )
    rows, mapping, adapter = parse_rows(raw, ".csv", offline=True)
    assert adapter == "heuristic"
    result = validate_statement(*normalise_rows(rows, mapping, business_id=1))
    assert [row.direction for row in result.unique_rows] == ["inflow", "outflow"]
    assert result.report["balance_reconciliation"]["passed"]
    monkeypatch.setattr(
        "app.services.ingest.parsers.propose_mapping", lambda *_: {"date": "missing"}
    )
    _, source = map_columns(list(rows[0]), rows, offline=False)
    assert source == "heuristic"


def test_edited_balance_is_flagged():
    raw = (
        b"date,description,inflow_pkr,outflow_pkr,balance_pkr\n"
        b"2026-09-01,Sales,500,0,1500\n"
        b"2026-09-02,Rent,0,200,9999\n"
        b"2026-09-03,Sales,100,0,1400\n"
    )
    rows, mapping, _ = parse_rows(raw, ".csv")
    result = validate_statement(*normalise_rows(rows, mapping, business_id=1))
    assert result.report["balance_reconciliation"]["mismatches"] == 2
    assert result.report["data_quality_score"] < 100


def test_corrupt_pdf_has_safe_failure_report():
    document = ingest_statement(1, b"%PDF-1.4\nnot a pdf", "account_PK36MEZN0001928374650101.pdf")
    assert document.parse_status.value == "failed"
    assert document.parse_report["errors"][0]["code"] == "PARSE_ERROR"
    assert "PK36" not in str(document.parse_report)
    assert "PK36" not in document.filename


def test_upload_status_sample_and_assurance():
    client = TestClient(app)
    path = SAMPLES / "ayeshas_boutique_statement.pdf"
    uploaded = client.post(
        "/api/v1/applications/1/documents",
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
    )
    assert uploaded.status_code == 201
    body = uploaded.json()
    assert body["status"] == "parsed"
    assert body["progress"] == {
        name: "complete" for name in ("parse", "categorise", "validate", "features")
    }
    assert client.get(f"/api/v1/applications/1/documents/{body['document_id']}").status_code == 200
    assert client.post("/api/v1/applications/2/documents/use-sample?format=csv").status_code == 200
    assert score_application(1)["assurance"]["data_quality"]["statement_quality_score"] == 100


def test_llm_mapping_is_validated_and_only_ambiguous_rows_are_sent(monkeypatch):
    from app.services.ingest import normalise, parsers

    raw = (
        b"Transaction Date,Particulars,Transaction Amount,Running Balance\n"
        b"2026-09-01,Unclear payment ZXQ,500,1500\n"
    )
    rows, mapping, _ = parse_rows(raw, ".csv")
    proposal = {
        "date": "Transaction Date",
        "description": "Particulars",
        "amount": "Transaction Amount",
        "balance": "Running Balance",
    }
    monkeypatch.setattr(parsers, "propose_mapping", lambda *_: proposal)
    _, source = map_columns(list(rows[0]), rows, offline=False)
    assert source == "llm_proposal"
    calls = []

    def fake_categories(batch):
        calls.append(batch)
        return {"Unclear payment ZXQ": {"category": "sales", "confidence": 0.82}}

    monkeypatch.setattr(normalise, "propose_categories", fake_categories)
    first, _ = normalise_rows(rows, mapping, business_id=1, offline=False)
    second, _ = normalise_rows(rows, mapping, business_id=1, offline=False)
    assert len(calls) == 1
    assert first[0].category == second[0].category == "sales"
    assert first[0].category_source == "llm"
    assert first[0].category_confidence == 0.82


def test_llm_bridge_masks_account_identifiers(monkeypatch):
    from app.services.ingest import llm_bridge

    captured = {}

    class Provider:
        def propose_statement_mapping(self, headers, examples):
            captured["headers"] = headers
            captured["examples"] = examples
            return {}

    monkeypatch.setattr(llm_bridge, "_provider", lambda: Provider())
    iban = "PK36MEZN0001928374650101"
    llm_bridge.propose_mapping(["date", iban], [{"date": "2026-09-01", iban: iban}])
    assert iban not in str(captured)
    assert "[ACCOUNT]" in str(captured)


def test_upload_limits_and_allow_list(monkeypatch):
    from app.core.config import settings

    client = TestClient(app)
    monkeypatch.setattr(settings, "MAX_STATEMENT_BYTES", 10)
    too_large = client.post(
        "/api/v1/applications/1/documents", files={"file": ("statement.csv", b"a" * 11, "text/csv")}
    )
    assert too_large.status_code == 413
    monkeypatch.setattr(settings, "MAX_STATEMENT_BYTES", 10_000_000)
    unsupported = client.post(
        "/api/v1/applications/1/documents",
        files={"file": ("statement.exe", b"hello", "application/octet-stream")},
    )
    assert unsupported.status_code == 415
