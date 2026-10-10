"""Statement upload, progress status and synthetic sample shortcut."""

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Application, Document
from app.db.session import get_db
from app.services.ingest.parsers import ParseError
from app.services.ingest.service import ingest_statement

router = APIRouter(tags=["documents"])
SAMPLES = Path(__file__).resolve().parents[4] / "samples"
PERSONA_SAMPLES = {1: "ayeshas_boutique", 2: "karim_traders", 3: "fastfix_logistics"}


class DocumentStatus(BaseModel):
    application_id: int
    document_id: int
    status: Literal["pending", "processing", "parsed", "failed"]
    filename: str
    sha256: str
    progress: dict[str, str]
    parse_report: dict[str, Any]


def _status(application_id: int, document: Document) -> DocumentStatus:
    report = document.parse_report or {}
    return DocumentStatus(
        application_id=application_id,
        document_id=document.id,
        status=document.parse_status.value,
        filename=document.filename,
        sha256=document.sha256,
        progress=report.get("steps", {}),
        parse_report=report,
    )


@router.post(
    "/applications/{application_id}/documents", response_model=DocumentStatus, status_code=201
)
async def upload_document(
    application_id: int, file: Annotated[UploadFile, File()]
) -> DocumentStatus:
    raw = await file.read(settings.MAX_STATEMENT_BYTES + 1)
    try:
        document = ingest_statement(application_id, raw, file.filename or "statement")
        return _status(application_id, document)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ParseError as exc:
        code = 413 if "exceeds" in str(exc) else 415
        raise HTTPException(code, str(exc)) from exc
    finally:
        await file.close()


@router.get("/applications/{application_id}/documents/{document_id}", response_model=DocumentStatus)
def document_status(
    application_id: int, document_id: int, session: Annotated[Session, Depends(get_db)]
) -> DocumentStatus:
    app = session.get(Application, application_id)
    document = session.get(Document, document_id)
    if app is None or document is None or document.business_id != app.business_id:
        raise HTTPException(404, "Document not found for this application")
    return _status(application_id, document)


@router.post("/applications/{application_id}/documents/use-sample", response_model=DocumentStatus)
def use_sample(
    application_id: int,
    format: Literal["csv", "pdf"] = "csv",
    session: Annotated[Session, Depends(get_db)] = None,
) -> DocumentStatus:
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(404, "Application not found")
    slug = PERSONA_SAMPLES.get(app.business_id)
    if slug is None:
        raise HTTPException(404, "No synthetic sample is available for this business")
    path = SAMPLES / f"{slug}_statement.{format}"
    document = ingest_statement(application_id, path.read_bytes(), path.name, synthetic_sample=True)
    return _status(application_id, document)
