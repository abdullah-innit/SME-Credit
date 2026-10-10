"""Prospective Path to Approval and fast what-if routes."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ml.counterfactual import ScenarioChanges, find_path_to_approval, what_if
from app.services.policy import PolicyCheck

router = APIRouter(tags=["path-to-approval"])


class ScorePreview(BaseModel):
    pd: float
    score: int
    tier: Literal["A", "B", "C", "D", "E"]
    decision: Literal["APPROVE", "REFER", "DECLINE"]
    recommended_amount_pkr: int | None = None


class ReasonContribution(BaseModel):
    feature: str
    value: float
    reference_value: float
    contribution: float
    unit: str
    display_en: str
    display_ur: str
    text_en: str
    text_ur: str


class WhatIfResponse(BaseModel):
    application_id: int
    changes: dict[str, float | int | bool]
    original: ScorePreview
    what_if: ScorePreview
    reason_changes: dict[str, list[str]]
    top_reasons_before: list[ReasonContribution]
    top_reasons_after: list[ReasonContribution]
    policy_checks: list[PolicyCheck]
    prospective_only: bool


class PlanStep(BaseModel):
    action: str
    feature: str
    from_value: float
    to_value: float
    unit: str
    effort_weight: float
    time_to_effect_months: int
    milestone: str
    text_en: str
    text_ur: str


class LenderView(BaseModel):
    policy_accepted: bool
    policy_checks: list[PolicyCheck]


class PathPlan(BaseModel):
    label: Literal["Fastest", "Cheapest", "Strongest"]
    changes: dict[str, float | int | bool]
    steps: list[PlanStep]
    timeline: dict[str, int | list[str]]
    total_effort: float
    predicted_pd: float
    pd_change: float
    predicted_score: int
    score_change: int
    expected_approved_amount_pkr: int
    lender_view: LenderView
    prospective_only: bool


class PathResponse(BaseModel):
    application_id: int
    original_decision: Literal["APPROVE", "REFER", "DECLINE"]
    original_pd: float | None = None
    original_score: int | None = None
    plans: list[PathPlan]
    blockers: list[str]
    lender_view: bool
    model_version: str | None = None
    policy_version: str | None = None


@router.post("/applications/{application_id}/what-if", response_model=WhatIfResponse)
def what_if_route(application_id: int, changes: ScenarioChanges) -> WhatIfResponse:
    try:
        return WhatIfResponse.model_validate(what_if(application_id, changes))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/applications/{application_id}/path-to-approval", response_model=PathResponse)
def path_route(application_id: int, lender_view: bool = False) -> PathResponse:
    try:
        return PathResponse.model_validate(
            find_path_to_approval(application_id, lender_view=lender_view)
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
