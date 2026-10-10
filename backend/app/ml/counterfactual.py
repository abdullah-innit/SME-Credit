"""Actionable, policy-verified Path to Approval scenarios for synthetic SMEs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Application, Business, FeatureSnapshot, ScoreResult
from app.db.session import SessionLocal
from app.ml.explain import explain_values
from app.ml.registry import FEATURES
from app.ml.scorecard import calibrated_probability, pd_to_score, tier_for_pd
from app.services.pipeline import _models, score_application
from app.services.policy import evaluate_policy, load_policy

PD_MARGIN = 0.005

# Sliders represent prospective evidence, never changes to the historical record.
ACTION_FEATURE = {
    "amount_requested_pkr": "amount_to_annual_revenue",
    "tenor_months": "instalment_to_inflow",
    "avg_balance_pkr": "avg_balance_ratio",
    "digital_payment_share": "digital_payment_share",
    "on_time_bill_ratio": "on_time_bill_ratio",
    "has_ntn": "has_ntn",
    "collateral_value_pkr": "collateral_coverage",
    "avg_monthly_net_cash_flow_pkr": "avg_monthly_net_cash_flow",
}

ACTION_UNIT = {
    "amount_requested_pkr": "PKR",
    "tenor_months": "months",
    "avg_balance_pkr": "PKR",
    "digital_payment_share": "share",
    "on_time_bill_ratio": "share",
    "has_ntn": "flag",
    "collateral_value_pkr": "PKR",
    "avg_monthly_net_cash_flow_pkr": "PKR/month",
}
ACTION_TEXT = {
    "amount_requested_pkr": (
        "Request PKR {to:,.0f} instead of PKR {from_:,.0f}.",
        "درخواست کردہ رقم {from_:,.0f} روپے سے کم کر کے {to:,.0f} روپے کریں۔",
    ),
    "tenor_months": (
        "Spread repayment over {to:.0f} months instead of {from_:.0f}.",
        "واپسی کی مدت {from_:.0f} سے بڑھا کر {to:.0f} ماہ کریں۔",
    ),
    "avg_balance_pkr": (
        "Maintain an average balance of PKR {to:,.0f}, up from PKR {from_:,.0f}.",
        "اوسط بقایا {from_:,.0f} روپے سے بڑھا کر {to:,.0f} روپے رکھیں۔",
    ),
    "digital_payment_share": (
        "Route {to:.0%} of sales digitally, up from {from_:.0%}.",
        "ڈیجیٹل فروخت کا حصہ {from_:.0%} سے بڑھا کر {to:.0%} کریں۔",
    ),
    "on_time_bill_ratio": (
        "Pay bills on time for three months: {from_:.0%} to {to:.0%}.",
        "تین ماہ بل بروقت ادا کر کے تناسب {from_:.0%} سے {to:.0%} کریں۔",
    ),
    "has_ntn": ("Register for an NTN.", "این ٹی این کے لیے رجسٹر ہوں۔"),
    "collateral_value_pkr": (
        "Add verifiable collateral from PKR {from_:,.0f} to PKR {to:,.0f}.",
        "قابل تصدیق ضمانت {from_:,.0f} روپے سے {to:,.0f} روپے کریں۔",
    ),
    "avg_monthly_net_cash_flow_pkr": (
        "Sustain cost savings: monthly net cash flow PKR {from_:,.0f} to {to:,.0f}.",
        "اخراجات گھٹا کر ماہانہ خالص نقد بہاؤ {from_:,.0f} سے {to:,.0f} روپے کریں۔",
    ),
}


class ScenarioChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_requested_pkr: int | None = Field(default=None, gt=0)
    tenor_months: int | None = Field(default=None, gt=0, le=60)
    avg_balance_pkr: float | None = Field(default=None, ge=0)
    digital_payment_share: float | None = Field(default=None, ge=0, le=1)
    on_time_bill_ratio: float | None = Field(default=None, ge=0, le=1)
    has_ntn: bool | None = None
    collateral_value_pkr: float | None = Field(default=None, ge=0)
    avg_monthly_net_cash_flow_pkr: float | None = None

    @model_validator(mode="after")
    def actionable_only(self) -> ScenarioChanges:
        for key in self.model_fields_set:
            if key not in ACTION_FEATURE or not FEATURES[ACTION_FEATURE[key]].actionable:
                raise ValueError(f"Slider {key} is not actionable")
        return self


@dataclass(frozen=True)
class ScenarioResult:
    changes: dict[str, float | int | bool]
    values: dict[str, float]
    pd: float
    challenger_pd: float
    score: int
    tier: str
    outcome: str
    amount_pkr: int
    policy_checks: tuple[dict[str, Any], ...]


def _original_slider_values(base: dict[str, float], app: Application) -> dict[str, float]:
    inflow = base["avg_monthly_inflow"]
    return {
        "amount_requested_pkr": float(app.amount_requested_pkr),
        "tenor_months": float(app.tenor_months),
        "avg_balance_pkr": base["avg_balance_ratio"] * inflow,
        "digital_payment_share": base["digital_payment_share"],
        "on_time_bill_ratio": base["on_time_bill_ratio"],
        "has_ntn": base["has_ntn"],
        "collateral_value_pkr": base["collateral_coverage"] * app.amount_requested_pkr,
        "avg_monthly_net_cash_flow_pkr": base["avg_monthly_net_cash_flow"],
    }


def project_features(
    base: dict[str, float], app: Application, changes: ScenarioChanges
) -> dict[str, float]:
    """Recompute dependent ratios from a prospective action set, without touching evidence."""
    changed = changes.model_dump(exclude_none=True)
    original = _original_slider_values(base, app)
    for key, target in changed.items():
        spec = FEATURES[ACTION_FEATURE[key]]
        if not spec.actionable:
            raise ValueError(f"{key} cannot change")
        if key == "on_time_bill_ratio" and not base["has_bill_history"]:
            raise ValueError("Bill-payment history is required for this action")
        if key == "avg_monthly_net_cash_flow_pkr" and target > base["avg_monthly_inflow"]:
            raise ValueError("Net cash flow cannot exceed inflow")
        if key == "avg_balance_pkr" and base["avg_monthly_inflow"] <= 0:
            raise ValueError("Balance ratio needs recorded inflow")
        if key == "digital_payment_share" and not base["has_transaction_history"]:
            raise ValueError("Digital share needs recorded sales")
    values = dict(base)
    amount = int(changed.get("amount_requested_pkr", app.amount_requested_pkr))
    tenor = int(changed.get("tenor_months", app.tenor_months))
    inflow = base["avg_monthly_inflow"]
    net_cash = float(
        changed.get("avg_monthly_net_cash_flow_pkr", original["avg_monthly_net_cash_flow_pkr"])
    )
    values["avg_monthly_net_cash_flow"] = net_cash
    old_instalment = app.amount_requested_pkr / app.tenor_months
    old_dscr = base["dscr_proxy"]
    known_service = (
        max(0.0, base["avg_monthly_net_cash_flow"] / old_dscr - old_instalment)
        if base["has_debt_service_data"] and old_dscr > 0
        else 0.0
    )
    instalment = amount / tenor
    denominator = instalment + known_service
    values["dscr_proxy"] = net_cash / denominator if denominator > 0 else 0.0
    values["instalment_to_inflow"] = instalment / inflow if inflow > 0 else 0.0
    values["amount_to_annual_revenue"] = amount / (12 * inflow) if inflow > 0 else 0.0
    collateral = float(changed.get("collateral_value_pkr", original["collateral_value_pkr"]))
    values["collateral_coverage"] = collateral / amount
    if "avg_balance_pkr" in changed:
        ratio = float(changed["avg_balance_pkr"]) / inflow
        values["avg_balance_ratio"] = ratio
        values["min_balance_ratio"] = ratio
    if "digital_payment_share" in changed:
        share = float(changed["digital_payment_share"])
        values["digital_payment_share"] = share
        values["cash_deposit_share"] = 1 - share
    if "on_time_bill_ratio" in changed:
        ratio = float(changed["on_time_bill_ratio"])
        old_ratio = base["on_time_bill_ratio"]
        values["on_time_bill_ratio"] = ratio
        values["avg_days_late"] = (
            max(0.0, base["avg_days_late"] * (1 - ratio) / (1 - old_ratio))
            if old_ratio < 1
            else 0.0
        )
    if "has_ntn" in changed:
        values["has_ntn"] = float(bool(changed["has_ntn"]))
    for name, value in values.items():
        low, high = FEATURES[name].bounds
        if not low <= value <= high:
            raise ValueError(f"Projected {name} lies outside realistic bounds")
    return values


def _batch_score(
    base: dict[str, float],
    app: Application,
    business: Business,
    candidates: list[dict[str, float | int | bool]],
    policy: dict,
) -> list[ScenarioResult]:
    champion, challenger, metadata = _models()
    projected = [project_features(base, app, ScenarioChanges(**c)) for c in candidates]
    matrix = np.asarray(
        [[v[name] for name in metadata["feature_list"]] for v in projected], dtype=float
    )
    pds = calibrated_probability(champion, matrix)
    challenger_pds = calibrated_probability(challenger, matrix)
    result = []
    for changes, values, pd, challenger_pd in zip(
        candidates, projected, pds, challenger_pds, strict=True
    ):
        pd = float(pd)
        challenger_pd = float(challenger_pd)
        tier = tier_for_pd(pd, metadata["scorecard"])
        decision = evaluate_policy(
            policy=policy,
            values=values,
            sector=business.sector,
            requested_amount_pkr=int(changes.get("amount_requested_pkr", app.amount_requested_pkr)),
            requested_tenor_months=int(changes.get("tenor_months", app.tenor_months)),
            tier=tier,
            champion_pd=pd,
            challenger_pd=challenger_pd,
        )
        result.append(
            ScenarioResult(
                changes,
                values,
                pd,
                challenger_pd,
                pd_to_score(pd, metadata["scorecard"]),
                tier,
                decision.outcome,
                decision.recommended_amount_pkr,
                tuple(asdict(check) for check in decision.checks),
            )
        )
    return result


def _context(
    application_id: int, session: Session
) -> tuple[Application, Business, dict[str, float]]:
    score_application(application_id, session=session)
    app = session.get(Application, application_id)
    if app is None:
        raise LookupError("Application not found")
    business = session.get(Business, app.business_id)
    score = session.scalar(
        select(ScoreResult)
        .where(ScoreResult.application_id == application_id)
        .order_by(ScoreResult.id.desc())
        .limit(1)
    )
    snapshot = session.get(FeatureSnapshot, score.feature_snapshot_id)
    return app, business, snapshot.values


def what_if(
    application_id: int, changes: ScenarioChanges, *, session: Session | None = None
) -> dict:
    own = session is None
    session = session or SessionLocal()
    try:
        app, business, base = _context(application_id, session)
        policy = load_policy()
        original, updated = _batch_score(
            base, app, business, [{}, changes.model_dump(exclude_none=True)], policy
        )
        champion, _, metadata = _models()
        baseline_reasons = explain_values(base, loaded=(champion, metadata), check_stability=False)
        updated_reasons = explain_values(
            updated.values, loaded=(champion, metadata), check_stability=False
        )
        before = {item.feature for item in baseline_reasons.top_risk_increasing}
        after = {item.feature for item in updated_reasons.top_risk_increasing}
        return {
            "application_id": application_id,
            "changes": updated.changes,
            "original": {
                "pd": original.pd,
                "score": original.score,
                "tier": original.tier,
                "decision": original.outcome,
            },
            "what_if": {
                "pd": updated.pd,
                "score": updated.score,
                "tier": updated.tier,
                "decision": updated.outcome,
                "recommended_amount_pkr": updated.amount_pkr,
            },
            "reason_changes": {"added": sorted(after - before), "removed": sorted(before - after)},
            "top_reasons_before": [asdict(item) for item in baseline_reasons.top_risk_increasing],
            "top_reasons_after": [asdict(item) for item in updated_reasons.top_risk_increasing],
            "policy_checks": list(updated.policy_checks),
            "prospective_only": True,
        }
    finally:
        if own:
            session.close()


def _grid(
    base: dict[str, float], app: Application, policy: dict
) -> dict[str, list[float | int | bool]]:
    original = _original_slider_values(base, app)
    amount = app.amount_requested_pkr
    inflow = base["avg_monthly_inflow"]
    minimum = int(policy["minimum_recommended_amount_pkr"])
    current_cap = max(
        int(limits["maximum_exposure_pkr"]) for limits in policy["tier_limits"].values()
    )
    affordable = int(
        max(
            0,
            min(
                base["avg_monthly_net_cash_flow"] / policy["dscr_floor"],
                inflow * policy["maximum_instalment_to_inflow"],
            ),
        )
        * app.tenor_months
    )
    amount_levels = {int(amount * ratio) for ratio in (0.9, 0.75, 0.6, 0.45)}
    amount_levels.update({current_cap, int(current_cap * 0.9), int(affordable * 0.98)})
    grid: dict[str, list[float | int | bool]] = {
        "amount_requested_pkr": sorted(
            {max(minimum, a) for a in amount_levels if minimum <= a < amount}, reverse=True
        ),
        "tenor_months": [t for t in (18, 24) if app.tenor_months < t],
    }
    if inflow > 0 and base["has_balance_data"]:
        grid["avg_balance_pkr"] = [
            round(original["avg_balance_pkr"] + inflow * ratio) for ratio in (0.1, 0.25, 0.5)
        ]
    if base["has_transaction_history"]:
        grid["digital_payment_share"] = [
            round(s, 2) for s in (0.5, 0.7, 0.9) if s > original["digital_payment_share"] + 0.01
        ]
    if base["has_bill_history"]:
        grid["on_time_bill_ratio"] = [
            s for s in (0.9, 1.0) if s > base["on_time_bill_ratio"] + 0.01
        ]
    if not base["has_ntn"]:
        grid["has_ntn"] = [True]
    grid["collateral_value_pkr"] = [
        round(original["collateral_value_pkr"] + amount * ratio) for ratio in (0.5, 1.5)
    ]
    grid["avg_monthly_net_cash_flow_pkr"] = [
        round(min(inflow, original["avg_monthly_net_cash_flow_pkr"] + inflow * ratio))
        for ratio in (0.1, 0.2)
        if original["avg_monthly_net_cash_flow_pkr"] + inflow * ratio <= inflow
    ]
    return {key: list(dict.fromkeys(levels)) for key, levels in grid.items() if levels}


def _effort(changes: dict[str, float | int | bool], original: dict[str, float]) -> float:
    total = 0.0
    for key, target in changes.items():
        spec = FEATURES[ACTION_FEATURE[key]]
        start = original[key]
        if key == "has_ntn":
            fraction = 1.0
        elif key in {
            "amount_requested_pkr",
            "collateral_value_pkr",
            "avg_balance_pkr",
            "avg_monthly_net_cash_flow_pkr",
        }:
            fraction = abs(float(target) - start) / max(1.0, abs(start), 100000.0)
        elif key == "tenor_months":
            fraction = abs(float(target) - start) / 12
        else:
            fraction = abs(float(target) - start)
        total += spec.effort_weight * max(0.05, fraction)
    return round(total, 4)


def _timeline_months(changes: dict[str, float | int | bool]) -> int:
    return max(
        (
            max(3, FEATURES[ACTION_FEATURE[key]].time_to_effect_months)
            if key == "on_time_bill_ratio"
            else FEATURES[ACTION_FEATURE[key]].time_to_effect_months
            for key in changes
        ),
        default=0,
    )


def _steps(changes: dict[str, float | int | bool], original: dict[str, float]) -> list[dict]:
    steps = []
    for key, target in changes.items():
        spec = FEATURES[ACTION_FEATURE[key]]
        months = (
            max(3, spec.time_to_effect_months)
            if key == "on_time_bill_ratio"
            else spec.time_to_effect_months
        )
        en, ur = ACTION_TEXT[key]
        start = original[key]
        steps.append(
            {
                "action": key,
                "feature": spec.name,
                "from_value": start,
                "to_value": float(target),
                "unit": ACTION_UNIT[key],
                "effort_weight": spec.effort_weight,
                "time_to_effect_months": months,
                "milestone": f"Month {months}" if months else "Week 1",
                "text_en": en.format(from_=start, to=float(target)),
                "text_ur": ur.format(from_=start, to=float(target)),
            }
        )
    return sorted(steps, key=lambda item: (item["time_to_effect_months"], item["action"]))


def _plan(
    label: str,
    candidate: ScenarioResult,
    original_score: ScenarioResult,
    original: dict[str, float],
) -> dict:
    months = _timeline_months(candidate.changes)
    return {
        "label": label,
        "changes": candidate.changes,
        "steps": _steps(candidate.changes, original),
        "timeline": {
            "duration_months": months,
            "milestones": [step["milestone"] for step in _steps(candidate.changes, original)],
        },
        "total_effort": _effort(candidate.changes, original),
        "predicted_pd": candidate.pd,
        "pd_change": round(candidate.pd - original_score.pd, 6),
        "predicted_score": candidate.score,
        "score_change": candidate.score - original_score.score,
        "expected_approved_amount_pkr": candidate.amount_pkr,
        "lender_view": {
            "policy_accepted": candidate.outcome == "APPROVE",
            "policy_checks": list(candidate.policy_checks),
        },
        "prospective_only": True,
    }


def find_path_to_approval(
    application_id: int, *, session: Session | None = None, lender_view: bool = False
) -> dict:
    """Bounded beam search; every returned plan is rescored by the real policy."""
    own = session is None
    session = session or SessionLocal()
    try:
        app, business, base = _context(application_id, session)
        policy = load_policy()
        original = _batch_score(base, app, business, [{}], policy)[0]
        if original.outcome == "APPROVE":
            return {
                "application_id": application_id,
                "original_decision": original.outcome,
                "plans": [],
                "blockers": [],
                "lender_view": lender_view,
            }
        hard = [
            check["code"]
            for check in original.policy_checks
            if check["status"] == "fail" and check["code"] in {"MIN_VINTAGE", "SECTOR_EXCLUSION"}
        ]
        if hard:
            return {
                "application_id": application_id,
                "original_decision": original.outcome,
                "plans": [],
                "blockers": hard,
                "lender_view": lender_view,
            }
        slider_start = _original_slider_values(base, app)
        grid = _grid(base, app, policy)
        beam: list[dict[str, float | int | bool]] = [{}]
        feasible: dict[tuple, ScenarioResult] = {}
        seen: set[tuple] = {()}
        for _depth in range(4):
            proposed = []
            for state in beam:
                for key, levels in grid.items():
                    if key in state:
                        continue
                    for level in levels:
                        next_state = {**state, key: level}
                        fingerprint = tuple(sorted(next_state.items()))
                        if fingerprint not in seen:
                            seen.add(fingerprint)
                            proposed.append(next_state)
            if not proposed:
                break
            valid = []
            for item in proposed:
                try:
                    project_features(base, app, ScenarioChanges(**item))
                    valid.append(item)
                except ValueError:
                    continue
            if not valid:
                break
            evaluated = _batch_score(base, app, business, valid, policy)
            ranked = []
            for candidate in evaluated:
                gap = abs(candidate.pd - candidate.challenger_pd)
                if candidate.outcome == "APPROVE" and candidate.pd <= 0.1 - PD_MARGIN:
                    feasible[tuple(sorted(candidate.changes.items()))] = candidate
                failed = sum(check["status"] == "fail" for check in candidate.policy_checks)
                priority = (
                    failed * 2
                    + max(0, candidate.pd - (0.1 - PD_MARGIN)) * 20
                    + max(0, gap - policy["maximum_champion_challenger_pd_gap"]) * 5
                    + _effort(candidate.changes, slider_start) * 0.03
                )
                ranked.append((priority, candidate))
            ranked.sort(key=lambda item: (item[0], _effort(item[1].changes, slider_start)))
            beam = [item.changes for _, item in ranked[:12]]
        candidates = list(feasible.values())
        chosen: list[dict] = []
        selectors = [
            (
                "Cheapest",
                lambda c: (_effort(c.changes, slider_start), _timeline_months(c.changes), c.pd),
            ),
            (
                "Fastest",
                lambda c: (_timeline_months(c.changes), _effort(c.changes, slider_start), c.pd),
            ),
            (
                "Strongest",
                lambda c: (c.pd, _effort(c.changes, slider_start), _timeline_months(c.changes)),
            ),
        ]
        for label, selector in selectors:
            available = [
                c
                for c in candidates
                if tuple(sorted(c.changes.items()))
                not in {tuple(sorted(p["changes"].items())) for p in chosen}
            ]
            if available:
                chosen.append(_plan(label, min(available, key=selector), original, slider_start))
        chosen.sort(key=lambda plan: {"Fastest": 0, "Cheapest": 1, "Strongest": 2}[plan["label"]])
        row = session.scalar(
            select(ScoreResult)
            .where(ScoreResult.application_id == application_id)
            .order_by(ScoreResult.id.desc())
            .limit(1)
        )
        if row is not None:
            row.counterfactuals = chosen
            session.flush()
        if own:
            session.commit()
        return {
            "application_id": application_id,
            "original_decision": original.outcome,
            "original_pd": original.pd,
            "original_score": original.score,
            "plans": chosen,
            "blockers": [],
            "lender_view": lender_view,
            "model_version": _models()[2]["model_version"],
            "policy_version": policy["effective_version"],
        }
    finally:
        if own:
            session.close()
