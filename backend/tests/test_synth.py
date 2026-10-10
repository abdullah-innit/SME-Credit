"""Tests for the synthetic data generator and label leakage sanity checks."""

import sqlite3

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from app.ml.synth import SyntheticDataGenerator


@pytest.fixture(scope="module")
def synth_data(tmp_path_factory):
    """Generate a small synthetic dataset for testing."""
    test_dir = tmp_path_factory.mktemp("synth_test")
    db_path = test_dir / "sme_credit_test.db"
    data_dir = test_dir / "data"
    samples_dir = test_dir / "samples"

    generator = SyntheticDataGenerator(n=300, seed=42)
    summary = generator.generate_all(
        db_path=db_path,
        data_dir=data_dir,
        samples_dir=samples_dir,
        quiet=True
    )

    return {
        "summary": summary,
        "db_path": db_path,
        "data_dir": data_dir,
        "samples_dir": samples_dir,
        "generator": generator
    }


def test_generator_summary_stats(synth_data):
    """Check if the generator produces stats within the expected ranges."""
    summary = synth_data["summary"]

    assert summary["n"] == 300
    assert 0.15 <= summary["women_led_share"] <= 0.25
    assert 0.25 <= summary["thin_file_share"] <= 0.45
    assert 0.50 <= summary["ntn_share"] <= 0.70
    assert 0.05 <= summary["default_rate"] <= 0.15
    assert 0.01 <= summary["fraud_share"] <= 0.08
    assert summary["balance_reconciliation_pct"] == 100.0


def test_label_leakage_and_schema_isolation(synth_data):
    """
    Sanity check: Ensure evaluation columns (default, fraud flags, latent health)
    are strictly isolated in the loan_evaluations table and do not leak into
    model feature tables (e.g., businesses, applications).
    """
    db_path = synth_data["db_path"]
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 1. Schema isolation
    cursor = conn.execute("PRAGMA table_info(businesses)")
    biz_cols = [row["name"] for row in cursor.fetchall()]

    forbidden_cols = ["default_12m", "is_synthetic_fraud", "fraud_pattern", "latent_health"]
    for col in forbidden_cols:
        assert col not in biz_cols, f"Leakage: '{col}' found in businesses table"

    cursor = conn.execute("PRAGMA table_info(applications)")
    app_cols = [row["name"] for row in cursor.fetchall()]
    for col in forbidden_cols:
        assert col not in app_cols, f"Leakage: '{col}' found in applications table"

    cursor = conn.execute("PRAGMA table_info(loan_evaluations)")
    eval_cols = [row["name"] for row in cursor.fetchall()]
    for col in ["default_12m", "is_synthetic_fraud", "latent_business_health", "fraud_pattern"]:
        assert col in eval_cols, f"Missing '{col}' in loan_evaluations table"

    # 2. No perfect separability check (Sanity Check ML Model)
    # Train a logistic regression on basic demographics to ensure AUC is reasonable (not 1.0)
    # This prevents accidental deterministic label generation based purely on features.

    # Fetch data
    query = """
        SELECT
            b.sector, b.city, b.legal_form, b.employees, b.owner_age_band,
            e.default_12m
        FROM businesses b
        JOIN loan_evaluations e ON b.id = e.business_id
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    assert len(rows) == 300

    # Extract features and labels
    X_cat = [[row["sector"], row["city"], row["legal_form"], row["owner_age_band"]] for row in rows]
    X_num = np.array([[row["employees"]] for row in rows])
    y = np.array([row["default_12m"] for row in rows])

    # Ensure we have both classes
    assert len(np.unique(y)) == 2, "Need both defaults and non-defaults for AUC test"

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Combine features
    X = np.hstack([X_cat_encoded, X_num])

    # Fit simple model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Predict probabilities
    y_pred_proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)

    # AUC should be better than random but not perfect.
    # With just basic demographics, it should be relatively weak (0.5 to 0.7)
    # This proves the labels aren't perfectly leaked through demographic features.
    assert auc < 0.95, f"Suspiciously high AUC ({auc:.3f}). Possible label leakage."
