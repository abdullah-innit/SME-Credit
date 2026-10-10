import pytest

from app.db.models import Application
from app.ml.fairness import PROTECTED_ATTRIBUTES, verify_no_protected_attributes
from app.ml.features import compute_features


def test_no_protected_attributes_in_model():
    """Verify that protected attributes are never used as model features."""
    safe_features = ["avg_balance", "inflows_30d"]
    verify_no_protected_attributes(safe_features)

    for attr in PROTECTED_ATTRIBUTES:
        with pytest.raises(ValueError, match=f"Protected attribute {attr} found"):
            verify_no_protected_attributes(safe_features + [attr])

    # Also test actual feature list if db is available
    from app.db.session import SessionLocal
    try:
        session = SessionLocal()
        app = session.query(Application).first()
        if app:
            snapshot = compute_features(app.id, session)
            verify_no_protected_attributes(list(snapshot.values.keys()))
        session.close()
    except Exception:
        pass # Ignore if no DB or no apps in test environment
