"""Calibrated PD loader and officer-facing scorecard."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import logit

from app.ml.baseline import load_policy
from app.ml.registry import FEATURE_VERSION, FEATURES, MODEL_FEATURES

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class RiskScore:
    pd: float
    score: int
    tier: str
    model_version: str
    feature_version: str
    adverse_reasons: tuple[str, ...]


def calibrated_probability(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    estimator = bundle["estimator"]
    probability = estimator.predict_proba(x)[:, 1]
    if bundle["method"] == "isotonic":
        result = bundle["calibrator"].predict(probability)
    elif bundle["method"] == "platt":
        if hasattr(estimator, "decision_function"):
            raw = estimator.decision_function(x)
        else:
            raw = logit(np.clip(probability, 1e-6, 1 - 1e-6))
        result = bundle["calibrator"].predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {bundle['method']}")
    return np.clip(np.asarray(result, dtype=float), 1e-6, 1 - 1e-6)


def load_model(artifact_dir: Path = ARTIFACTS) -> tuple[dict, dict]:
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata missing at {metadata_path}; run python -m app.ml.train"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = load_policy()["model_version"]
    if metadata.get("model_version") != expected:
        raise ValueError(
            f"Model version mismatch: expected {expected}, got {metadata.get('model_version')}"
        )
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Feature version mismatch; retrain the model")
    if metadata.get("feature_list") != list(MODEL_FEATURES):
        raise ValueError("Feature list mismatch; retrain the model")
    path = artifact_dir / f"{metadata['champion']}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact missing at {path}; run python -m app.ml.train")
    return joblib.load(path), metadata


def pd_to_score(pd: float, card: Mapping[str, Any]) -> int:
    """Base 600 at 50:1 good-to-bad odds; +20 points per odds doubling."""
    probability = min(1 - 1e-6, max(1e-6, pd))
    odds = (1 - probability) / probability
    points = card["base_score"] + card["points_to_double_odds"] * math.log2(
        odds / card["base_good_to_bad_odds"]
    )
    return round(min(card["max_score"], max(card["min_score"], points)))


def tier_for_pd(pd: float, card: Mapping[str, Any]) -> str:
    return next(tier for tier, upper in card["tier_pd_upper"].items() if pd < upper or tier == "E")


def score_values(values: Mapping[str, float], loaded: tuple[dict, dict] | None = None) -> RiskScore:
    bundle, metadata = loaded or load_model()
    if set(values) != set(metadata["feature_list"]):
        raise ValueError("Scoring values do not match trained feature list")
    x = np.array([[values[name] for name in metadata["feature_list"]]], dtype=float)
    pd = float(calibrated_probability(bundle, x)[0])
    card = metadata["scorecard"]
    score = pd_to_score(pd, card)
    tier = tier_for_pd(pd, card)
    estimator = bundle["estimator"]
    scaler = estimator.named_steps["standardscaler"]
    classifier = estimator.named_steps["logisticregression"]
    contribution = scaler.transform(x)[0] * classifier.coef_[0]
    ranked = sorted(enumerate(contribution), key=lambda item: item[1], reverse=True)
    reasons = []
    for i, value in ranked:
        name = metadata["feature_list"][i]
        direction = FEATURES[name].monotonic
        coefficient = classifier.coef_[0][i]
        if value > 0 and (
            (direction == "better" and coefficient < 0)
            or (direction == "worse" and coefficient > 0)
        ):
            reasons.append(FEATURES[name].display_en)
        if len(reasons) == 3:
            break
    return RiskScore(
        pd, score, tier, metadata["model_version"], metadata["feature_version"], tuple(reasons)
    )


def score_application(application_id: int) -> RiskScore:
    """Score current evidence and persist the full bilingual explanation."""
    from app.ml.explain import explain_application

    explanation = explain_application(application_id)
    return RiskScore(
        explanation.calibrated_pd,
        explanation.score,
        explanation.tier,
        explanation.model_version,
        explanation.feature_version,
        tuple(item.display_en for item in explanation.top_risk_increasing[:3]),
    )
