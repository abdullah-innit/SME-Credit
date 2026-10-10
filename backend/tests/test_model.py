"""Training split, artifact and scorecard acceptance tests."""

import json
import time
from pathlib import Path

import pytest

from app.ml.baseline import load_policy, traditional_baseline
from app.ml.features import compute_features
from app.ml.scorecard import ARTIFACTS, load_model, pd_to_score, score_values, tier_for_pd
from app.ml.train import split_businesses


def test_business_split_is_disjoint_and_stratified():
    records = [
        {"business_id": i, "default_12m": int(i % 10 == 0)} for i in range(100) for _ in range(2)
    ]
    train, validation, test = split_businesses(records, load_policy())
    assert len(train) == 60
    assert len(validation) == 20
    assert len(test) == 20
    assert not (
        set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test)
    )
    assert sum(i % 10 == 0 for i in train) == 6
    assert sum(i % 10 == 0 for i in validation) == 2
    assert sum(i % 10 == 0 for i in test) == 2


def test_artifact_loader_and_monotonic_drivers(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="run python -m app.ml.train"):
        load_model(tmp_path)
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_version"] = "invalid"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        load_model(tmp_path)
    loaded = load_model()
    values = compute_features(2).values
    assert (
        score_values(values | {"bureau_score": 300}, loaded).pd
        > score_values(values | {"bureau_score": 850}, loaded).pd
    )
    assert (
        score_values(values | {"on_time_bill_ratio": 0}, loaded).pd
        > score_values(values | {"on_time_bill_ratio": 1}, loaded).pd
    )
    assert (
        score_values(values | {"instalment_to_inflow": 1.5}, loaded).pd
        > score_values(values | {"instalment_to_inflow": 0.05}, loaded).pd
    )
    score = score_values(values, loaded)
    assert 300 <= score.score <= 900
    assert score.tier in "ABCDE"
    assert score.adverse_reasons
    assert 0.6 < metadata["metrics"]["models"]["logistic"]["test"]["all"]["roc_auc"] < 0.95
    # Warm the runtime, then include disk artifact load and scoring in the timed call.
    score_values(values)
    start = time.perf_counter()
    score_values(values)
    assert time.perf_counter() - start < 0.1


def test_scorecard_anchor_and_baseline():
    cfg = load_policy()
    card = cfg["scorecard"]
    odds = card["base_good_to_bad_odds"]
    import math

    points = card["base_score"] + card["points_to_double_odds"] * math.log2(odds / odds)
    doubled = card["base_score"] + card["points_to_double_odds"] * math.log2(2 * odds / odds)
    assert points == 600
    assert doubled - points == 20
    assert pd_to_score(1 / 51, card) == 600
    assert pd_to_score(1 / 101, card) == 620
    assert tier_for_pd(0.019, card) == "A"
    assert tier_for_pd(0.20, card) == "E"
    assert traditional_baseline(bureau_history_months=24, collateral_coverage=1.5).approve
    assert not traditional_baseline(bureau_history_months=23, collateral_coverage=1.5).approve
    assert not traditional_baseline(bureau_history_months=24, collateral_coverage=1.49).approve
