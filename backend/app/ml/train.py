# ruff: noqa: E501
"""Train and validate synthetic SME default-risk models.

Run from backend: python -m app.ml.train
Labels are read only from loan_evaluations after point-in-time features are made.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from app.db.session import SessionLocal
from app.ml.baseline import load_policy, traditional_baseline
from app.ml.features import compute_features
from app.ml.registry import FEATURE_VERSION, MODEL_FEATURES, validate_registry
from app.ml.scorecard import calibrated_probability

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def _records(session: Any) -> list[dict[str, Any]]:
    # Evaluation-only fields are deliberately retrieved in this training module.
    query = text("""
        SELECT e.application_id, e.business_id, e.default_12m, e.is_thin_file,
               COALESCE(MAX(br.history_months), 0) AS bureau_history_months
        FROM loan_evaluations e
        JOIN applications a ON a.id = e.application_id
        LEFT JOIN bureau_records br ON br.business_id = e.business_id
            AND julianday(br.created_at) <= julianday(COALESCE(a.submitted_at, a.created_at))
        GROUP BY e.id
        ORDER BY e.business_id, e.application_id
    """)
    return [dict(row) for row in session.execute(query).mappings()]


def split_businesses(
    records: list[dict[str, Any]],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify unique businesses and keep every application of a business together."""
    labels: dict[int, int] = {}
    for row in records:
        business_id = int(row["business_id"])
        label = int(row["default_12m"])
        if business_id in labels and labels[business_id] != label:
            raise ValueError("A business has conflicting default labels")
        labels[business_id] = label
    ids = np.array(sorted(labels))
    y = np.array([labels[i] for i in ids])
    cfg = config["split"]
    if not math.isclose(sum(cfg.values()), 1.0):
        raise ValueError("Split shares must sum to one")
    train_ids, hold_ids = train_test_split(
        ids,
        test_size=cfg["validation"] + cfg["test"],
        stratify=y,
        random_state=config["random_seed"],
    )
    hold_labels = np.array([labels[i] for i in hold_ids])
    val_ids, test_ids = train_test_split(
        hold_ids,
        test_size=cfg["test"] / (cfg["validation"] + cfg["test"]),
        stratify=hold_labels,
        random_state=config["random_seed"] + 1,
    )
    return train_ids, val_ids, test_ids


def _raw_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    return logit(np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6))


def _fit_calibration(model: Any, x: np.ndarray, y: np.ndarray) -> tuple[str, Any, float]:
    raw = _raw_score(model, x)
    probability = model.predict_proba(x)[:, 1]
    sigmoid = LogisticRegression(C=1e3, max_iter=1000).fit(raw.reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(probability, y)
    candidates = {
        "platt": (sigmoid, sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1]),
        "isotonic": (isotonic, isotonic.predict(probability)),
    }
    method = min(candidates, key=lambda k: brier_score_loss(y, candidates[k][1]))
    calibrator, predictions = candidates[method]
    return method, calibrator, float(brier_score_loss(y, predictions))


def _table(y: np.ndarray, p: np.ndarray) -> list[dict[str, float | int]]:
    order = np.argsort(-p)
    chunks = np.array_split(order, 10)
    return [
        {
            "decile": i + 1,
            "n": int(len(chunk)),
            "mean_pd": float(np.mean(p[chunk])),
            "observed_default_rate": float(np.mean(y[chunk])),
        }
        for i, chunk in enumerate(chunks)
    ]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": int(len(y)),
        "defaults": int(sum(y)),
        "default_rate": float(np.mean(y)) if len(y) else None,
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        result.update(
            {
                "roc_auc": None,
                "pr_auc": None,
                "ks": None,
                "brier": None,
                "calibration_table": [],
                "lift_by_decile": [],
            }
        )
        return result
    fpr, tpr, _ = roc_curve(y, p)
    result.update(
        {
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "ks": float(np.max(tpr - fpr)),
            "brier": float(brier_score_loss(y, p)),
        }
    )
    table = _table(y, p)
    result["calibration_table"] = table
    result["lift_by_decile"] = [
        {"decile": row["decile"], "lift": row["observed_default_rate"] / result["default_rate"]}
        for row in table
    ]
    return result


def _markdown(metrics: dict, config: dict, data_hash: str) -> tuple[str, str]:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    rows = []
    for model_name, data in metrics["models"].items():
        for group, summary in data["test"].items():
            rows.append(
                f"| {model_name} | {group} | {summary['n']} | {summary['defaults']} | "
                f"{fmt(summary['roc_auc'])} | {fmt(summary['pr_auc'])} | "
                f"{fmt(summary['ks'])} | {fmt(summary['brier'])} |"
            )
    validation = f"""# Synthetic model validation

Generated: {metrics["trained_at"]} UTC. Data SHA-256: `{data_hash}`.
This is a demonstration on synthetic businesses and synthetic defaults, not a claim of real-world performance.

## Split and selection

Business-disjoint stratified split with seed {config["random_seed"]}: train {metrics["split"]["train"]}, validation {metrics["split"]["validation"]}, test {metrics["split"]["test"]}. Standardization is fit on train only. Both estimators are fit on train only. Platt and isotonic calibrators are fit on validation; validation Brier chooses each calibrator. Logistic regression is the prespecified champion; the boosting model is a challenger. The test set is evaluated once after selection. Validation Brier is optimistic because it also selects the calibrator.

Champion: **{metrics["champion"]}**. Champion validation Brier: {fmt(metrics["selection_brier"])}.

| Model | Test segment | N | Defaults | ROC-AUC | PR-AUC | KS | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Champion calibration by risk decile

| Risk decile | N | Mean predicted PD | Observed default rate | Lift |
|---:|---:|---:|---:|---:|
"""
    champion = metrics["models"][metrics["champion"]]["test"]["all"]
    for row, lift in zip(champion["calibration_table"], champion["lift_by_decile"], strict=True):
        validation += (
            f"| {row['decile']} | {row['n']} | {fmt(row['mean_pd'])} | "
            f"{fmt(row['observed_default_rate'])} | {fmt(lift['lift'])} |\n"
        )
    base = metrics["baseline"]
    validation += f"""
## Traditional-rule baseline on the same test businesses

Requires at least {config["baseline"]["min_bureau_history_months"]} bureau-history months and collateral coverage at least {config["baseline"]["min_collateral_coverage"]}×. Approval rate: {fmt(base["approval_rate"])}; observed default rate among approved: {fmt(base["approved_default_rate"])} ({base["approved"]} approved). These are policy outcomes, not model discrimination metrics.
"""
    card = f"""# Synthetic default-risk model card

Version: {config["model_version"]}; feature version: {FEATURE_VERSION}. Trained: {metrics["trained_at"]} UTC.

## Intended use

Decision support for synthetic SME loan demonstrations. The model estimates a 12-month probability of default from point-in-time cash-flow, bill, bureau, and application features. Officers can review the score, evidence, and top adverse feature associations relative to the training mean. These reasons are not causal claims. A human officer retains the final decision; an LLM never decides credit.

## Model and score

Champion: {metrics["champion"]}. The bank-friendly logistic candidate uses standardized features and L2 regularization; the challenger is a histogram gradient boosting classifier. The selected model is calibrated on validation. Score 600 corresponds to good-to-bad odds of 50:1; doubling those odds adds 20 points. Scores are clipped to 300–900. PD tiers: A <2%, B <5%, C <10%, D <20%, E ≥20%.

## Data, evaluation, and limits

All 3,000 businesses, statements, bureau records, and defaults are synthetic. Synthetic default labels are generated from latent business health and injected fraud; they do not validate performance on real borrowers. Business-disjoint train/validation/test partitions prevent entity leakage. Owner gender and age band are excluded from model features and reserved for fairness audit. Thin-file and thick-file test results appear in [VALIDATION.md](VALIDATION.md).

The DSCR proxy lacks observed existing-loan payments and can overstate capacity. Profile fields are unversioned; historical profile edits cannot be reconstructed. The six-month transaction window may miss annual seasonality. Calibration and tier thresholds require real-world validation and governance before any lending use. Missing bureau data is explicitly flagged; a score of 600 is an imputation, not a real bureau score.
"""
    return validation, card


def train() -> dict[str, Any]:
    config = load_policy()
    validate_registry(MODEL_FEATURES)
    with SessionLocal() as session:
        records = _records(session)
        if not records:
            raise RuntimeError("No evaluation labels found; generate synthetic data first")
        train_ids, val_ids, test_ids = split_businesses(records, config)
        assignment = {
            int(i): name
            for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids))
            for i in ids
        }
        # The snapshot is created before its label enters any estimator.
        features = [compute_features(int(r["application_id"]), session).values for r in records]
        session.commit()
    x = np.array([[f[name] for name in MODEL_FEATURES] for f in features], dtype=float)
    y = np.array([int(r["default_12m"]) for r in records], dtype=int)
    groups = np.array([assignment[int(r["business_id"])] for r in records])
    thin = np.array([bool(r["is_thin_file"]) for r in records])
    digest = hashlib.sha256()
    for row, feature in zip(records, features, strict=True):
        digest.update(
            json.dumps(
                {
                    "application_id": row["application_id"],
                    "business_id": row["business_id"],
                    "label": row["default_12m"],
                    "features": feature,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    data_hash = digest.hexdigest()
    estimators = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=1000, random_state=config["random_seed"]),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=config["random_seed"],
        ),
    }
    bundles = {}
    selection = {}
    for name, model in estimators.items():
        model.fit(x[groups == "train"], y[groups == "train"])
        method, calibrator, brier = _fit_calibration(
            model, x[groups == "validation"], y[groups == "validation"]
        )
        bundles[name] = {"estimator": model, "calibrator": calibrator, "method": method}
        selection[name] = brier
    champion = "logistic"
    train_reference = np.mean(x[groups == "train"], axis=0)
    logistic_coefficients = estimators["logistic"].named_steps["logisticregression"].coef_[0]
    challenger_importance = permutation_importance(
        estimators["hist_gradient_boosting"],
        x[groups == "validation"],
        y[groups == "validation"],
        scoring="roc_auc",
        n_repeats=3,
        random_state=config["random_seed"],
    ).importances_mean
    global_importance = {
        "logistic": {
            name: float(abs(coefficient))
            for name, coefficient in zip(MODEL_FEATURES, logistic_coefficients, strict=True)
        },
        "hist_gradient_boosting": {
            name: float(max(0, importance))
            for name, importance in zip(MODEL_FEATURES, challenger_importance, strict=True)
        },
    }
    reports = {}
    for name, bundle in bundles.items():
        mask = groups == "test"
        p = calibrated_probability(bundle, x[mask])
        reports[name] = {
            "calibration": bundle["method"],
            "validation_brier": selection[name],
            "test": {
                "all": _metrics(y[mask], p),
                "thin_file": _metrics(y[mask & thin], p[thin[mask]]),
                "thick_file": _metrics(y[mask & ~thin], p[~thin[mask]]),
            },
        }
    baseline_mask = groups == "test"
    approved = np.array(
        [
            traditional_baseline(
                bureau_history_months=int(r["bureau_history_months"]),
                collateral_coverage=features[i]["collateral_coverage"],
                config=config,
            ).approve
            for i, r in enumerate(records)
            if baseline_mask[i]
        ]
    )
    test_y = y[baseline_mask]
    baseline = {
        "approved": int(sum(approved)),
        "approval_rate": float(np.mean(approved)),
        "approved_default_rate": float(np.mean(test_y[approved])) if any(approved) else None,
    }
    now = datetime.now(UTC).isoformat()
    metrics = {
        "trained_at": now,
        "champion": champion,
        "selection_brier": selection[champion],
        "split": {key: int(sum(groups == key)) for key in ("train", "validation", "test")},
        "models": reports,
        "baseline": baseline,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    for name, bundle in bundles.items():
        joblib.dump(bundle, ARTIFACTS / f"{name}.joblib")
    metadata = {
        "model_version": config["model_version"],
        "feature_version": FEATURE_VERSION,
        "feature_list": list(MODEL_FEATURES),
        "data_hash_sha256": data_hash,
        "metrics": metrics,
        "trained_at": now,
        "champion": champion,
        "scorecard": config["scorecard"],
        "training_reference": train_reference.tolist(),
        "global_importance": global_importance,
        "split_business_ids": {
            "train": train_ids.tolist(),
            "validation": val_ids.tolist(),
            "test": test_ids.tolist(),
        },
    }
    (ARTIFACTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    validation, card = _markdown(metrics, config, data_hash)
    (ROOT / "docs" / "VALIDATION.md").write_text(validation, encoding="utf-8")
    (ROOT / "docs" / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = train()
    summary = result["metrics"]["models"][result["champion"]]["test"]["all"]
    print(
        f"Champion {result['champion']}: test ROC-AUC={summary['roc_auc']:.3f}, "
        f"PR-AUC={summary['pr_auc']:.3f}, Brier={summary['brier']:.3f}"
    )
