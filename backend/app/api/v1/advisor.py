"""Advisor routes for Applicant Assistant and Officer Copilot."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.knowledge.store import get_knowledge_store
from app.services.advisor import (
    AdvisorRequest,
    AdvisorResponse,
    advisor_engine,
    session_memory,
    stream_advisor_chat,
)

router = APIRouter(prefix="/advisor", tags=["advisor"])


@router.post("/chat", response_model=AdvisorResponse)
def advisor_chat(
    req: AdvisorRequest,
    db: Annotated[Session, Depends(get_db)],
) -> AdvisorResponse:
    """Non-streaming conversational advisor query.

    Handles applicant assistant and officer copilot personas with tool execution,
    guardrails, polite refusals, and chunk citations.
    """
    try:
        return advisor_engine.process(req, session=db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Advisor processing error: {exc}") from exc


@router.post("/chat/stream")
def advisor_chat_stream(
    req: AdvisorRequest,
    db: Annotated[Session, Depends(get_db)],
) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming endpoint for real-time conversation."""
    return StreamingResponse(
        stream_advisor_chat(req, session=db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}")
def get_session_history(session_id: str) -> dict[str, Any]:
    """Retrieve turn history for an active advisor session."""
    turns = session_memory.get_history(session_id)
    return {
        "session_id": session_id,
        "turn_count": len(turns),
        "history": [
            {
                "role": t.role,
                "message": t.message,
                "timestamp": t.timestamp.isoformat(),
                "tool_calls": [tc.model_dump() for tc in t.tool_calls],
                "citations": t.citations,
            }
            for t in turns
        ],
    }


@router.delete("/history/{session_id}")
def clear_session_history(session_id: str) -> dict[str, str]:
    """Clear memory turns for an active advisor session."""
    session_memory.clear(session_id)
    return {"session_id": session_id, "status": "cleared"}


@router.get("/knowledge")
def list_knowledge(
    category: str | None = Query(default=None, description="Optional category filter"),
    query: str | None = Query(default=None, description="Optional search query"),
) -> dict[str, Any]:
    """Retrieve knowledge base chunks or search policy documents."""
    store = get_knowledge_store()
    if query:
        retriever = advisor_engine.retriever
        matches = retriever.search(query=query, category=category, top_k=5)
        return {
            "query": query,
            "count": len(matches),
            "results": [m.as_dict() for m in matches],
        }

    chunks = store.list_chunks(category=category)
    return {
        "count": len(chunks),
        "chunks": [c.as_dict() for c in chunks],
    }
