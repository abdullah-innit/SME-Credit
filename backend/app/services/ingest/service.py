"""Persist parsed statements without retaining raw account data."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import enums
from app.core.config import settings
from app.db.models import Application, AuditLog, Document, Transaction
from app.db.session import SessionLocal
from app.ml.features import compute_features
from app.services.ingest.normalise import _digest, normalise_rows
from app.services.ingest.parsers import ParseError, parse_rows
from app.services.ingest.validate import validate_statement

ALLOWED_SUFFIXES = frozenset({".csv", ".xlsx", ".pdf"})
STEPS = ("parse", "categorise", "validate", "features")


def _initial_report(*, synthetic_sample: bool) -> dict:
    return {
        "steps": {name: "pending" for name in STEPS},
        "data_quality_score": 0,
        "source": "synthetic_sample" if synthetic_sample else "user_upload",
    }


def _check_type(raw: bytes, suffix: str) -> None:
    if suffix not in ALLOWED_SUFFIXES:
        raise ParseError("Upload a CSV, XLSX or text-based PDF statement")
    if suffix == ".pdf" and not raw.startswith(b"%PDF"):
        raise ParseError("File is not a valid PDF")
    if suffix == ".xlsx" and not raw.startswith(b"PK\x03\x04"):
        raise ParseError("File is not a valid XLSX workbook")
    if suffix == ".csv" and b"\x00" in raw[:4096]:
        raise ParseError("CSV appears to contain binary data")


def _existing_signatures(session: Session, business_id: int, start, end) -> set[str]:
    rows = session.scalars(
        select(Transaction).where(
            Transaction.business_id == business_id,
            Transaction.date >= start,
            Transaction.date <= end,
        )
    ).all()
    return {
        _digest(
            (
                row.date,
                row.amount_pkr,
                row.direction.value,
                row.channel.value,
                row.balance_after if row.balance_after is not None else row.counterparty_hash,
            )
        )
        for row in rows
    }


def ingest_statement(
    application_id: int,
    raw: bytes,
    filename: str,
    *,
    session: Session | None = None,
    synthetic_sample: bool = False,
) -> Document:
    """Synchronous stages are durable in API mode; malformed files get FAILED reports."""
    if len(raw) > settings.MAX_STATEMENT_BYTES:
        raise ParseError(
            f"Statement exceeds the {settings.MAX_STATEMENT_BYTES // 1_000_000} MB limit"
        )
    suffix = Path(filename).suffix.lower()
    _check_type(raw, suffix)
    own = session is None
    session = session or SessionLocal()
    try:
        app = session.get(Application, application_id)
        if app is None:
            raise LookupError("Application not found")
        digest = hashlib.sha256(raw).hexdigest()
        existing = session.scalar(
            select(Document)
            .where(
                Document.business_id == app.business_id,
                Document.sha256 == digest,
                Document.parse_status.in_([enums.ParseStatus.PARSED, enums.ParseStatus.FAILED]),
            )
            .limit(1)
        )
        if existing is not None:
            return existing
        safe_name = f"statement_{digest[:12]}{suffix}"
        document = Document(
            business_id=app.business_id,
            type=enums.DocumentType.BANK_STATEMENT,
            filename=safe_name,
            sha256=digest,
            parse_status=enums.ParseStatus.PROCESSING,
            parse_report=_initial_report(synthetic_sample=synthetic_sample),
        )
        session.add(document)
        session.flush()
        if own:
            session.commit()
        if settings.INGEST_DEBUG_KEEP_RAW:
            debug_dir = Path(__file__).resolve().parents[3] / ".ingest_debug"
            debug_dir.mkdir(exist_ok=True)
            (debug_dir / safe_name).write_bytes(raw)
        report = _initial_report(synthetic_sample=synthetic_sample)
        try:
            rows, mapping, adapter = parse_rows(raw, suffix, offline=settings.OFFLINE_MODE)
            report["steps"]["parse"] = "complete"
            report["adapter"] = adapter
            report["raw_rows"] = len(rows)
            document.parse_report = dict(report)
            session.flush()
            if own:
                session.commit()
            normalised, row_errors = normalise_rows(
                rows, mapping, business_id=app.business_id, offline=settings.OFFLINE_MODE
            )
            report["steps"]["categorise"] = "complete"
            report["category_decisions"] = [
                {
                    "row_hash": row.row_hash,
                    "category": row.category,
                    "confidence": row.category_confidence,
                    "source": row.category_source,
                }
                for row in normalised
            ]
            report["category_source_counts"] = dict(
                Counter(row.category_source for row in normalised)
            )
            document.parse_report = dict(report)
            session.flush()
            if own:
                session.commit()
            validation = validate_statement(normalised, row_errors)
            report.update(validation.report)
            report["steps"]["validate"] = "complete"
            document.parse_report = dict(report)
            session.flush()
            if own:
                session.commit()
            unique = validation.unique_rows
            previous = _existing_signatures(
                session,
                app.business_id,
                min(row.date for row in unique),
                max(row.date for row in unique),
            )
            added = 0
            for row in unique:
                if row.row_hash in previous:
                    continue
                previous.add(row.row_hash)
                session.add(
                    Transaction(
                        business_id=app.business_id,
                        date=row.date,
                        amount_pkr=row.amount_pkr,
                        direction=enums.Direction(row.direction),
                        channel=enums.Channel(row.channel),
                        category=row.category,
                        counterparty_hash=row.counterparty_hash,
                        balance_after=row.balance_after,
                        document_id=document.id,
                    )
                )
                added += 1
            session.flush()
            report["inserted_transactions"] = added
            report["existing_duplicates"] = len(unique) - added
            feature_result = compute_features(application_id, session)
            report["feature_snapshot_id"] = feature_result.snapshot_id
            report["steps"]["features"] = "complete"
            document.parse_report = report
            document.parse_status = enums.ParseStatus.PARSED
            session.add(
                AuditLog(
                    actor=app.business.owner_id,
                    action="statement_ingested",
                    entity="documents",
                    entity_id=document.id,
                    before=None,
                    after={
                        "sha256": digest,
                        "inserted_transactions": added,
                        "data_quality_score": report["data_quality_score"],
                    },
                    at=datetime.now(UTC),
                )
            )
            session.flush()
            if own:
                session.commit()
            return document
        except Exception as exc:
            session.rollback()
            document = session.get(Document, document.id) if own else document
            if document is None:
                raise
            safe_message = (
                str(exc)
                if isinstance(exc, (ParseError, ValueError))
                else "Statement could not be processed"
            )
            report["steps"] = {
                name: (
                    "failed"
                    if name
                    == next((k for k, v in report["steps"].items() if v == "pending"), "parse")
                    else value
                )
                for name, value in report["steps"].items()
            }
            report["errors"] = [{"code": "PARSE_ERROR", "message": safe_message}]
            document.parse_report = report
            document.parse_status = enums.ParseStatus.FAILED
            session.add(document)
            if own:
                session.commit()
            return document
    finally:
        if own:
            session.close()
