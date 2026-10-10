"""Faithful local model attributions and bilingual, rule-based reasons.

Waterfall contributions are on the uncalibrated log-odds scale. The subsequent
isotonic/Platt calibration is nonlinear and is reported separately as PD.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import joblib
import numpy as np
from scipy.special import logit
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import enums
from app.db.models import Decision, ScoreResult
from app.db.session import SessionLocal
from app.ml.features import compute_features
from app.ml.registry import FEATURES, MODEL_FEATURES
from app.ml.scorecard import (
    ARTIFACTS,
    calibrated_probability,
    load_model,
    pd_to_score,
    tier_for_pd,
)

# Every model feature has a reviewed, bilingual plain-language phrase.
REASON_TEMPLATES: dict[str, tuple[str, str]] = {
    "avg_monthly_inflow": ("Average monthly receipts", "اوسط ماہانہ آمدنی"),
    "inflow_cv": ("Variation in monthly receipts", "ماہانہ آمدنی میں اتار چڑھاؤ"),
    "revenue_trend_slope": ("Six-month revenue trend", "چھ ماہ کی آمدنی کا رجحان"),
    "negative_cash_flow_months": ("Months with negative cash flow", "منفی نقد بہاؤ والے ماہ"),
    "avg_balance_ratio": (
        "Average account balance relative to receipts",
        "آمدنی کے مقابلے میں اوسط بقایا",
    ),
    "min_balance_ratio": (
        "Lowest account balance relative to receipts",
        "آمدنی کے مقابلے میں کم ترین بقایا",
    ),
    "dscr_proxy": ("Estimated debt-payment capacity", "قرض کی ادائیگی کی تخمینی گنجائش"),
    "instalment_to_inflow": (
        "Proposed instalment relative to receipts",
        "آمدنی کے مقابلے میں مجوزہ قسط",
    ),
    "amount_to_annual_revenue": (
        "Requested loan relative to annual revenue",
        "سالانہ آمدنی کے مقابلے میں مطلوبہ قرض",
    ),
    "top3_customer_concentration": (
        "Share from the three largest customers",
        "تین بڑے گاہکوں سے حاصل ہونے والی آمدنی کا حصہ",
    ),
    "digital_payment_share": (
        "Share of receipts paid digitally",
        "ڈیجیٹل طریقے سے موصولہ آمدنی کا حصہ",
    ),
    "cash_deposit_share": ("Share of receipts deposited as cash", "نقد جمع شدہ آمدنی کا حصہ"),
    "active_day_ratio": ("Share of days with account activity", "لین دین والے دنوں کا حصہ"),
    "seasonality_index": ("Seasonal variation in receipts", "آمدنی میں موسمی تبدیلی"),
    "on_time_bill_ratio": ("Share of bills paid on time", "بروقت ادا شدہ بلوں کا حصہ"),
    "avg_days_late": ("Average delay in paying bills", "بلوں کی ادائیگی میں اوسط تاخیر"),
    "vintage_months": ("Time in business", "کاروبار کا دورانیہ"),
    "has_ntn": ("Tax registration", "ٹیکس رجسٹریشن"),
    "bureau_score": ("Credit bureau score", "کریڈٹ بیورو اسکور"),
    "delinquency_flag": ("Reported overdue debt", "رپورٹ شدہ تاخیر زدہ قرض"),
    "active_loan_count": ("Number of active loans", "فعال قرضوں کی تعداد"),
    "employees": ("Reported employee count", "رپورٹ شدہ ملازمین کی تعداد"),
    "collateral_coverage": (
        "Collateral relative to requested loan",
        "مطلوبہ قرض کے مقابلے میں ضمانت",
    ),
    "has_bureau_record": ("Availability of a bureau record", "بیورو ریکارڈ کی دستیابی"),
    "has_transaction_history": (
        "Availability of transaction history",
        "لین دین کی تاریخ کی دستیابی",
    ),
    "has_bill_history": (
        "Availability of bill-payment history",
        "بلوں کی ادائیگی کی تاریخ کی دستیابی",
    ),
    "has_balance_data": ("Availability of balance data", "بقایا معلومات کی دستیابی"),
    "has_debt_service_data": (
        "Availability of existing debt-payment data",
        "موجودہ قرض کی ادائیگی کی معلومات",
    ),
    "transaction_count": ("Number of recent transactions", "حالیہ لین دین کی تعداد"),
    "avg_monthly_net_cash_flow": ("Average monthly net cash flow", "اوسط ماہانہ خالص نقد بہاؤ"),
}

if set(REASON_TEMPLATES) != set(MODEL_FEATURES):
    raise ValueError("Every model feature must have an English and Urdu reason template")


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    reference_value: float
    contribution: float
    unit: str
    display_en: str
    display_ur: str
    text_en: str
    text_ur: str


@dataclass(frozen=True)
class AdverseActionNotice:
    outcome: str
    principal_features: tuple[str, ...]
    text_en: str
    text_ur: str


@dataclass(frozen=True)
class Explanation:
    application_id: int | None
    score_result_id: int | None
    model_version: str
    feature_version: str
    model_name: str
    attribution_method: str
    attribution_scale: str
    baseline_log_odds: float
    prediction_log_odds: float
    calibrated_pd: float
    score: int
    tier: str
    top_risk_increasing: tuple[FeatureContribution, ...]
    top_risk_reducing: tuple[FeatureContribution, ...]
    waterfall: tuple[FeatureContribution, ...]
    global_importance: tuple[dict[str, Any], ...]
    reasons_stable: bool
    stability_agreement: float
    adverse_action_notice: AdverseActionNotice | None = None


def _raw_log_odds(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(matrix), dtype=float)
    return logit(np.clip(estimator.predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6))


def _permutation_attributions(
    estimator: Any,
    vector: np.ndarray,
    reference: np.ndarray,
    permutations: int = 32,
) -> tuple[float, float, np.ndarray, str]:
    """Path-permutation Shapley approximation; each path telescopes exactly."""
    n_features = len(vector)
    rng = np.random.default_rng(42)
    rows = np.repeat(reference.reshape(1, -1), permutations * (n_features + 1), axis=0)
    order = np.empty((permutations, n_features), dtype=int)
    for sample in range(permutations):
        order[sample] = rng.permutation(n_features)
        offset = sample * (n_features + 1)
        for step, feature in enumerate(order[sample], start=1):
            rows[offset + step] = rows[offset + step - 1]
            rows[offset + step, feature] = vector[feature]
    margins = _raw_log_odds(estimator, rows).reshape(permutations, n_features + 1)
    contributions = np.zeros(n_features)
    for sample in range(permutations):
        for step, feature in enumerate(order[sample]):
            contributions[feature] += margins[sample, step + 1] - margins[sample, step]
    contributions /= permutations
    return float(margins[0, 0]), float(margins[0, -1]), contributions, "permutation_shapley"


def _tree_attributions(
    estimator: Any,
    vector: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, np.ndarray, str]:
    try:
        import shap  # type: ignore[import-not-found]

        explainer = shap.TreeExplainer(estimator, data=reference.reshape(1, -1), model_output="raw")
        shap_values = explainer.shap_values(vector.reshape(1, -1))
        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, -1]
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1], dtype=float)
        contributions = values.reshape(-1)
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
        predicted = float(_raw_log_odds(estimator, vector.reshape(1, -1))[0])
        if len(contributions) == len(vector) and math.isclose(
            expected + sum(contributions), predicted, abs_tol=1e-3
        ):
            return expected, predicted, contributions, "shap_tree"
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return _permutation_attributions(estimator, vector, reference)


def _attributions(
    bundle: dict[str, Any],
    model_name: str,
    vector: np.ndarray,
    reference: np.ndarray,
    *,
    permutations: int = 32,
) -> tuple[float, float, np.ndarray, str]:
    estimator = bundle["estimator"]
    if model_name == "logistic":
        scaler = estimator.named_steps["standardscaler"]
        classifier = estimator.named_steps["logisticregression"]
        contributions = scaler.transform(vector.reshape(1, -1))[0] * classifier.coef_[0]
        baseline = float(classifier.intercept_[0])
        predicted = float(estimator.decision_function(vector.reshape(1, -1))[0])
        return baseline, predicted, contributions, "standardized_coefficient"
    if model_name == "hist_gradient_boosting":
        if permutations == 32:
            return _tree_attributions(estimator, vector, reference)
        return _permutation_attributions(estimator, vector, reference, permutations)
    raise ValueError(f"Unknown model: {model_name}")


def _render(feature: str, value: float, contribution: float) -> tuple[str, str]:
    en, ur = REASON_TEMPLATES[feature]
    if FEATURES[feature].unit == "flag":
        shown_en = "yes" if value >= 0.5 else "no"
        shown_ur = "ہاں" if value >= 0.5 else "نہیں"
    else:
        shown_en = f"{value:,.2f} {FEATURES[feature].unit}"
        shown_ur = f"{value:,.2f}"
    if contribution >= 0:
        return (
            f"{en} ({shown_en}) raised estimated risk relative to the training average.",
            f"{ur} ({shown_ur}) نے تربیتی اوسط کے مقابلے میں تخمینی خطرہ بڑھایا۔",
        )
    return (
        f"{en} ({shown_en}) reduced estimated risk relative to the training average.",
        f"{ur} ({shown_ur}) نے تربیتی اوسط کے مقابلے میں تخمینی خطرہ کم کیا۔",
    )


def _top_indices(
    contributions: np.ndarray, count: int = 5
) -> tuple[frozenset[int], frozenset[int]]:
    # Ignore contributions too small to be meaningful as reasons.
    threshold = max(0.01, 0.01 * float(np.max(np.abs(contributions))))
    increasing = tuple(i for i in np.argsort(-contributions) if contributions[i] > threshold)[
        :count
    ]
    reducing = tuple(i for i in np.argsort(contributions) if contributions[i] < -threshold)[:count]
    return frozenset(increasing), frozenset(reducing)


def _stability(
    bundle: dict[str, Any],
    model_name: str,
    vector: np.ndarray,
    reference: np.ndarray,
    base_contributions: np.ndarray,
) -> tuple[bool, float]:
    original = _top_indices(base_contributions)
    rng = np.random.default_rng(1729)
    continuous = np.array(
        [FEATURES[name].unit not in {"flag", "count", "months", "days"} for name in MODEL_FEATURES]
    )
    scale = np.maximum(np.abs(vector), np.maximum(np.abs(reference), 1.0))
    matches = 0
    for _ in range(8):
        perturbed = vector + rng.normal(0, 1e-6, len(vector)) * scale * continuous
        _, _, shifted, _ = _attributions(bundle, model_name, perturbed, reference, permutations=8)
        matches += _top_indices(shifted) == original
    agreement = matches / 8
    return agreement == 1.0, agreement


def explain_values(
    values: Mapping[str, float],
    *,
    application_id: int | None = None,
    model_name: str | None = None,
    loaded: tuple[dict, dict] | None = None,
    check_stability: bool = True,
) -> Explanation:
    bundle, metadata = loaded or load_model()
    model_name = model_name or metadata["champion"]
    if model_name != metadata["champion"]:
        path = ARTIFACTS / f"{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Challenger artifact missing: {path}")
        bundle = joblib.load(path)
    if set(values) != set(metadata["feature_list"]):
        raise ValueError("Feature values do not match trained model")
    names = metadata["feature_list"]
    vector = np.array([values[name] for name in names], dtype=float)
    reference = np.array(metadata["training_reference"], dtype=float)
    baseline, predicted, contributions, method = _attributions(
        bundle, model_name, vector, reference
    )
    if not math.isclose(baseline + float(sum(contributions)), predicted, abs_tol=1e-3):
        raise ValueError("Local attributions do not reconcile to model log odds")
    pd = float(calibrated_probability(bundle, vector.reshape(1, -1))[0])
    specs = []
    for name, value, contribution in zip(names, vector, contributions, strict=True):
        en, ur = _render(name, float(value), float(contribution))
        spec = FEATURES[name]
        specs.append(
            FeatureContribution(
                name,
                float(value),
                float(reference[len(specs)]),
                float(contribution),
                spec.unit,
                spec.display_en,
                spec.display_ur,
                en,
                ur,
            )
        )
    increasing = tuple(
        sorted(
            (item for item in specs if item.contribution > 0),
            key=lambda item: item.contribution,
            reverse=True,
        )[:5]
    )
    reducing = tuple(
        sorted(
            (item for item in specs if item.contribution < 0),
            key=lambda item: item.contribution,
        )[:5]
    )
    importance = tuple(
        {
            "feature": name,
            "importance": float(importance),
            "display_en": FEATURES[name].display_en,
            "display_ur": FEATURES[name].display_ur,
        }
        for name, importance in sorted(
            metadata["global_importance"][model_name].items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    stable, agreement = (
        _stability(bundle, model_name, vector, reference, contributions)
        if check_stability
        else (True, 1.0)
    )
    card = metadata["scorecard"]
    return Explanation(
        application_id,
        None,
        metadata["model_version"],
        metadata["feature_version"],
        model_name,
        method,
        "raw_log_odds",
        baseline,
        predicted,
        pd,
        pd_to_score(pd, card),
        tier_for_pd(pd, card),
        increasing,
        reducing,
        tuple(specs),
        importance,
        stable,
        agreement,
    )


def adverse_action_notice(explanation: Explanation, outcome: str) -> AdverseActionNotice:
    if outcome not in {"rejected", "referred"}:
        raise ValueError("Adverse-action notice requires rejected or referred outcome")
    # Only use directional features whose observed value points toward risk.
    reasons = [
        item
        for item in explanation.top_risk_increasing
        if (
            (FEATURES[item.feature].monotonic == "better" and item.value < item.reference_value)
            or (FEATURES[item.feature].monotonic == "worse" and item.value > item.reference_value)
        )
        and not (
            item.feature == "bureau_score"
            and next(
                (row.value for row in explanation.waterfall if row.feature == "has_bureau_record"),
                0,
            )
            == 0
        )
    ][:3]
    if not reasons:
        reasons = list(explanation.top_risk_increasing[:3])
    en_reasons = "; ".join(item.text_en for item in reasons)
    ur_reasons = "؛ ".join(item.text_ur for item in reasons)
    status_en = "declined" if outcome == "rejected" else "referred for review"
    status_ur = "مسترد" if outcome == "rejected" else "مزید جائزے کے لیے بھیجی گئی"
    return AdverseActionNotice(
        outcome,
        tuple(item.feature for item in reasons),
        f"Your application was {status_en}. Principal model reasons: {en_reasons} "
        "A credit officer must review the evidence. You may provide corrected records.",
        f"آپ کی درخواست {status_ur} ہے۔ ماڈل کی اہم وجوہات: {ur_reasons} "
        "کریڈٹ افسر شواہد کا جائزہ لے گا۔ آپ درست معلومات فراہم کر سکتے ہیں۔",
    )


def explain_application(application_id: int, session: Session | None = None) -> Explanation:
    """Compute current evidence, persist a new snapshot and ScoreResult atomically."""
    own_session = session is None
    if session is None:
        session = SessionLocal()
    try:
        snapshot = compute_features(application_id, session)
        explanation = explain_values(snapshot.values, application_id=application_id)
        latest_decision = session.scalar(
            select(Decision)
            .where(Decision.application_id == application_id)
            .order_by(Decision.id.desc())
            .limit(1)
        )
        if latest_decision is not None and latest_decision.outcome in {
            enums.Outcome.REJECT,
            enums.Outcome.REFER,
        }:
            outcome = "rejected" if latest_decision.outcome == enums.Outcome.REJECT else "referred"
            explanation = replace(
                explanation, adverse_action_notice=adverse_action_notice(explanation, outcome)
            )
        stored_tier = (
            enums.RiskTier.LOW
            if explanation.tier in {"A", "B"}
            else enums.RiskTier.MEDIUM
            if explanation.tier == "C"
            else enums.RiskTier.HIGH
        )
        record = ScoreResult(
            application_id=application_id,
            feature_snapshot_id=snapshot.snapshot_id,
            model_version=explanation.model_version,
            pd=explanation.calibrated_pd,
            score=explanation.score,
            tier=stored_tier,
            reasons=[
                asdict(item)
                for item in (*explanation.top_risk_increasing, *explanation.top_risk_reducing)
            ],
            counterfactuals=[],
        )
        session.add(record)
        session.flush()
        if own_session:
            session.commit()
        return replace(explanation, score_result_id=record.id)
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()
