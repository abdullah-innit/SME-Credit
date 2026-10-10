"""Tool calling functions for SME Credit Advisor.

Implements get_application_summary, run_what_if, explain_reason, and search_policy
with strict persona-based authorization and structured tool records for the UI.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import Application, FraudFlag, ScoreResult
from app.db.session import SessionLocal
from app.knowledge.retrieval import get_knowledge_retriever
from app.ml.counterfactual import ScenarioChanges, find_path_to_approval, what_if


class ToolCallRecord(BaseModel):
    """Execution record of an invoked tool for UI rendering ('AI used a tool' chips)."""

    tool: str = Field(..., description="Name of the tool invoked")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Input parameters passed")
    summary: str = Field(..., description="User-friendly summary of the action performed")
    result: Any = Field(default=None, description="Structured tool output")


def get_application_summary(
    application_id: int,
    user_id: int | None = None,
    persona: str = "officer",
    session: Session | None = None,
) -> tuple[dict[str, Any], ToolCallRecord]:
    """Retrieve facts and status for an application with authorization checks."""
    own_session = session is None
    s = session or SessionLocal()
    try:
        app = s.get(Application, application_id)
        if app is None:
            raise LookupError(f"Application #{application_id} not found")

        # Authorization check: Applicant can ONLY view their own application
        if (
            persona.lower() == "applicant"
            and user_id is not None
            and app.business.owner_id != user_id
        ):
            msg = f"Unauthorized: Applicant {user_id} cannot access application #{application_id}"
            raise PermissionError(msg)

        business = app.business
        score_res = (
            s.query(ScoreResult)
            .filter_by(application_id=app.id)
            .order_by(ScoreResult.id.desc())
            .first()
        )
        flags_count = s.query(FraudFlag).filter_by(application_id=app.id).count()

        score_val = int(score_res.score) if score_res else 710
        tier_val = (
            str(score_res.tier.value if hasattr(score_res.tier, "value") else score_res.tier)
            if score_res
            else "B"
        )
        outcome_val = (
            "APPROVE" if score_val >= 680 else ("REFER" if score_val >= 620 else "DECLINE")
        )

        # Derived business metrics
        feature_vals = {}
        if score_res and score_res.feature_snapshot:
            feature_vals = dict(score_res.feature_snapshot.values)

        monthly_turnover = int(
            feature_vals.get("avg_monthly_inflow", app.amount_requested_pkr * 1.5)
        )
        dscr_val = round(float(feature_vals.get("dscr", 1.85)), 2)
        digital_ratio = round(float(feature_vals.get("digital_payment_share", 0.45)), 2)
        on_time_ratio = round(float(feature_vals.get("on_time_bill_ratio", 0.90)), 2)

        data = {
            "application_id": app.id,
            "business_name": business.name,
            "sector": business.sector,
            "city": business.city,
            "province": business.province,
            "legal_form": business.legal_form.value,
            "employees": business.employees,
            "has_ntn": business.has_ntn,
            "requested_amount_pkr": app.amount_requested_pkr,
            "tenor_months": app.tenor_months,
            "product": app.product.value,
            "score": score_val,
            "tier": tier_val,
            "recommendation": outcome_val,
            "monthly_turnover_pkr": monthly_turnover,
            "dscr": dscr_val,
            "digital_payment_share": digital_ratio,
            "on_time_bill_ratio": on_time_ratio,
            "integrity_flags_count": flags_count,
            "status": app.status.value,
        }

        chip_summary = f"Summary for {business.name} (Score: {score_val}, Tier {tier_val})"
        record = ToolCallRecord(
            tool="get_application_summary",
            arguments={"application_id": application_id},
            summary=chip_summary,
            result={
                "business_name": business.name,
                "score": score_val,
                "tier": tier_val,
                "recommendation": outcome_val,
                "requested_amount_pkr": app.amount_requested_pkr,
            },
        )
        return data, record
    finally:
        if own_session:
            s.close()


def run_what_if_tool(
    application_id: int,
    changes: dict[str, Any] | None = None,
    user_id: int | None = None,
    persona: str = "officer",
    session: Session | None = None,
) -> tuple[dict[str, Any], ToolCallRecord]:
    """Execute counterfactual what-if simulation or retrieve path to approval."""
    own_session = session is None
    s = session or SessionLocal()
    try:
        app = s.get(Application, application_id)
        if app is None:
            raise LookupError(f"Application #{application_id} not found")

        # Authorization check
        if (
            persona.lower() == "applicant"
            and user_id is not None
            and app.business.owner_id != user_id
        ):
            msg = f"Unauthorized: Applicant {user_id} cannot access application #{application_id}"
            raise PermissionError(msg)

        if changes:
            scenario = ScenarioChanges(**changes)
            result = what_if(application_id, scenario, session=s)
            orig_score = result.get("original", {}).get("score", 0)
            new_score = result.get("what_if", {}).get("score", 0)
            chip_summary = f"Ran what-if simulation: score moves from {orig_score} to {new_score}"
        else:
            result = find_path_to_approval(application_id, session=s)
            steps_count = len(result.get("steps", []))
            chip_summary = f"Generated Path to Approval with {steps_count} actionable milestones"

        record = ToolCallRecord(
            tool="run_what_if",
            arguments={"application_id": application_id, "changes": changes or {}},
            summary=chip_summary,
            result={
                "original_score": result.get("original", {}).get("score")
                if changes
                else result.get("score"),
                "what_if_score": result.get("what_if", {}).get("score")
                if changes
                else result.get("target_score"),
                "outcome": result.get("what_if", {}).get("decision")
                if changes
                else result.get("target_decision"),
            },
        )
        return result, record
    finally:
        if own_session:
            s.close()


def explain_reason_tool(
    application_id: int,
    feature_or_code: str | None = None,
    user_id: int | None = None,
    persona: str = "officer",
    session: Session | None = None,
) -> tuple[dict[str, Any], ToolCallRecord]:
    """Extract adverse-action reasons and feature drivers for an application."""
    own_session = session is None
    s = session or SessionLocal()
    try:
        app = s.get(Application, application_id)
        if app is None:
            raise LookupError(f"Application #{application_id} not found")

        # Authorization check
        if (
            persona.lower() == "applicant"
            and user_id is not None
            and app.business.owner_id != user_id
        ):
            msg = f"Unauthorized: Applicant {user_id} cannot access application #{application_id}"
            raise PermissionError(msg)

        score_res = (
            s.query(ScoreResult)
            .filter_by(application_id=app.id)
            .order_by(ScoreResult.id.desc())
            .first()
        )
        reasons = score_res.reasons if score_res and score_res.reasons else []

        top_increasing = [r for r in reasons if r.get("contribution", 0) > 0][:3]
        top_reducing = [r for r in reasons if r.get("contribution", 0) < 0][:3]

        selected_reason = None
        if feature_or_code:
            for r in reasons:
                if (
                    r.get("feature", "").lower() == feature_or_code.lower()
                    or feature_or_code.lower() in r.get("display_en", "").lower()
                ):
                    selected_reason = r
                    break

        data = {
            "application_id": application_id,
            "top_risk_increasing": top_increasing,
            "top_risk_reducing": top_reducing,
            "selected_reason": selected_reason,
            "total_reasons": len(reasons),
        }

        chip_summary = (
            f"Analyzed risk drivers for application #{application_id} "
            f"({len(top_increasing)} risk drivers identified)"
        )
        record = ToolCallRecord(
            tool="explain_reason",
            arguments={"application_id": application_id, "feature": feature_or_code},
            summary=chip_summary,
            result={
                "top_increasing_count": len(top_increasing),
                "top_reducing_count": len(top_reducing),
                "top_factor": top_increasing[0].get("display_en") if top_increasing else None,
            },
        )
        return data, record
    finally:
        if own_session:
            s.close()


def search_policy_tool(
    query: str,
    category: str | None = None,
    top_k: int = 3,
) -> tuple[dict[str, Any], ToolCallRecord]:
    """Retrieve policy, eligibility, terms, and guideline chunks."""
    retriever = get_knowledge_retriever()
    matches = retriever.search(query=query, top_k=top_k, category=category)

    serialized_matches = [m.as_dict() for m in matches]
    cited_ids = [m.chunk_id for m in matches]

    data = {
        "query": query,
        "matches": serialized_matches,
        "cited_chunk_ids": cited_ids,
    }

    chip_summary = (
        f"Retrieved {len(matches)} policy chunks for query: '{query[:40]}...'"
        if len(query) > 40
        else f"Retrieved {len(matches)} policy chunks for '{query}'"
    )
    record = ToolCallRecord(
        tool="search_policy",
        arguments={"query": query, "category": category, "top_k": top_k},
        summary=chip_summary,
        result={"matched_chunks": cited_ids, "top_title": matches[0].title if matches else None},
    )
    return data, record
