"""Traditional bureau/collateral rule baseline for holdout comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model.yaml"


@dataclass(frozen=True)
class BaselineDecision:
    approve: bool
    reasons: tuple[str, ...]


def load_policy() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def traditional_baseline(
    *,
    bureau_history_months: int,
    collateral_coverage: float,
    config: dict | None = None,
) -> BaselineDecision:
    policy = (config or load_policy())["baseline"]
    reasons = []
    if bureau_history_months < policy["min_bureau_history_months"]:
        reasons.append("Insufficient bureau history")
    if collateral_coverage < policy["min_collateral_coverage"]:
        reasons.append("Insufficient collateral coverage")
    return BaselineDecision(not reasons, tuple(reasons))
