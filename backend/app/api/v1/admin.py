"""Admin endpoints for monitoring and usage tracking."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AiCallLog
from app.db.session import get_db
from app.llm.gateway import get_health_info

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/ai-health")
def get_ai_health() -> dict[str, Any]:
    """Return provider, mode (online or offline), and last success timestamp."""
    return get_health_info()


@router.get("/ai-usage")
def get_ai_usage(
    limit: int = Query(default=50, ge=1, le=500),
    task: str | None = None,
    db: Annotated[Session, Depends(get_db)] = None,
) -> dict[str, Any]:
    """Aggregate latency, tokens, cost, and recent call logs from AiCallLog."""
    query = select(AiCallLog)
    if task:
        query = query.where(AiCallLog.task == task)

    total_calls = db.scalar(select(func.count(AiCallLog.id))) or 0
    total_tokens = db.scalar(select(func.sum(AiCallLog.tokens))) or 0
    total_cost = db.scalar(select(func.sum(AiCallLog.cost_usd))) or 0.0
    avg_duration = db.scalar(select(func.avg(AiCallLog.duration_ms))) or 0.0

    by_status: dict[str, int] = {}
    for st, count in db.execute(
        select(AiCallLog.status, func.count(AiCallLog.id)).group_by(AiCallLog.status)
    ).all():
        val = st.value if hasattr(st, "value") else str(st)
        by_status[val] = count

    by_task: dict[str, int] = {}
    for tsk, count in db.execute(
        select(AiCallLog.task, func.count(AiCallLog.id)).group_by(AiCallLog.task)
    ).all():
        by_task[str(tsk)] = count

    by_provider: dict[str, int] = {}
    for prov, count in db.execute(
        select(AiCallLog.provider, func.count(AiCallLog.id)).group_by(AiCallLog.provider)
    ).all():
        by_provider[str(prov)] = count

    recent_rows = db.scalars(query.order_by(AiCallLog.id.desc()).limit(limit)).all()
    recent_logs = [
        {
            "id": r.id,
            "application_id": r.application_id,
            "actor": r.actor,
            "provider": r.provider,
            "model": r.model,
            "task": r.task,
            "status": r.status.value if hasattr(r.status, "value") else str(r.status),
            "duration_ms": r.duration_ms,
            "prompt_version": r.prompt_version,
            "tokens": r.tokens,
            "cost_usd": round(r.cost_usd, 6),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent_rows
    ]

    return {
        "total_calls": total_calls,
        "total_tokens": int(total_tokens),
        "total_cost_usd": round(float(total_cost), 6),
        "avg_duration_ms": round(float(avg_duration), 2),
        "by_status": by_status,
        "by_task": by_task,
        "by_provider": by_provider,
        "recent_logs": recent_logs,
    }
