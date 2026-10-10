"""Versioned policy checks and offer calculation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "policy.yaml"


@dataclass(frozen=True)
class PolicyCheck:
    code: str
    status: Literal["pass", "fail", "warn"]
    message: str
    observed: float | str | None
    threshold: float | str | None


@dataclass(frozen=True)
class PolicyResult:
    outcome: Literal["APPROVE", "REFER", "DECLINE"]
    recommended_amount_pkr: int
    tenor_months: int
    indicative_rate_band: str
    expected_loss_pkr: float
    checks: tuple[PolicyCheck, ...]


def load_policy(path: Path = POLICY_PATH) -> dict:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    required = {
        "version",
        "minimum_vintage_months",
        "excluded_sectors",
        "dscr_floor",
        "maximum_instalment_to_inflow",
        "maximum_champion_challenger_pd_gap",
        "lgd",
        "minimum_recommended_amount_pkr",
        "tier_limits",
    }
    if (
        not isinstance(data, dict)
        or required - data.keys()
        or set(data["tier_limits"]) != set("ABCDE")
    ):
        raise ValueError("Invalid policy configuration")
    if data["dscr_floor"] <= 0 or not 0 < data["lgd"] <= 1:
        raise ValueError("Invalid policy DSCR floor or LGD")
    data["effective_version"] = f"{data['version']}+{hashlib.sha256(raw).hexdigest()[:12]}"
    return data


def evaluate_policy(
    *,
    policy: dict,
    values: dict[str, float],
    sector: str,
    requested_amount_pkr: int,
    requested_tenor_months: int,
    tier: str,
    champion_pd: float,
    challenger_pd: float,
) -> PolicyResult:
    limit = policy["tier_limits"][tier]
    tenor = min(requested_tenor_months, int(limit["maximum_tenor_months"]))
    tier_cap = int(limit["maximum_exposure_pkr"])
    floor = float(policy["dscr_floor"])
    share = float(policy["maximum_instalment_to_inflow"])
    monthly_capacity = max(
        0.0, min(values["avg_monthly_net_cash_flow"] / floor, values["avg_monthly_inflow"] * share)
    )
    amount = min(requested_amount_pkr, int(monthly_capacity * tenor), tier_cap)
    checks: list[PolicyCheck] = []

    def add(
        code: str,
        ok: bool,
        message: str,
        observed: float | str,
        threshold: float | str,
        *,
        warn: bool = False,
    ) -> None:
        checks.append(
            PolicyCheck(
                code, "pass" if ok else ("warn" if warn else "fail"), message, observed, threshold
            )
        )

    add(
        "MIN_VINTAGE",
        values["vintage_months"] >= policy["minimum_vintage_months"],
        "Business history must meet the minimum.",
        values["vintage_months"],
        policy["minimum_vintage_months"],
    )
    add(
        "SECTOR_EXCLUSION",
        sector.casefold() not in {s.casefold() for s in policy["excluded_sectors"]},
        "Business sector must be eligible.",
        sector,
        "eligible sector",
    )
    add(
        "TIER_EXPOSURE",
        requested_amount_pkr <= tier_cap,
        "Requested amount exceeds the tier limit.",
        requested_amount_pkr,
        tier_cap,
    )
    add(
        "DSCR_FLOOR",
        values["dscr_proxy"] >= floor,
        "Estimated cash flow must cover the instalment.",
        round(values["dscr_proxy"], 3),
        floor,
    )
    add(
        "INSTALMENT_CAP",
        values["instalment_to_inflow"] <= share,
        "Instalment must fit monthly receipts.",
        round(values["instalment_to_inflow"], 3),
        share,
    )
    add(
        "TENOR_LIMIT",
        requested_tenor_months <= tenor,
        "Tenor may be shortened.",
        requested_tenor_months,
        tenor,
        warn=True,
    )
    gap = abs(champion_pd - challenger_pd)
    add(
        "MODEL_DISAGREEMENT",
        gap <= policy["maximum_champion_challenger_pd_gap"],
        "Model disagreement requires officer review.",
        round(gap, 4),
        policy["maximum_champion_challenger_pd_gap"],
        warn=True,
    )
    add(
        "DEBT_SERVICE_UNVERIFIED",
        bool(values["has_debt_service_data"]),
        "Existing debt instalments are unavailable; DSCR is a proxy.",
        values["has_debt_service_data"],
        1,
        warn=True,
    )
    add(
        "MIN_OFFER",
        amount >= policy["minimum_recommended_amount_pkr"],
        "Affordable offer must meet the minimum.",
        amount,
        policy["minimum_recommended_amount_pkr"],
    )
    if (
        any(c.status == "fail" for c in checks[:2])
        or amount < policy["minimum_recommended_amount_pkr"]
    ):
        outcome = "DECLINE"
    elif (
        tier in {"D", "E"}
        or any(c.status == "fail" for c in checks)
        or gap > policy["maximum_champion_challenger_pd_gap"]
    ):
        outcome = "REFER"
    else:
        outcome = "APPROVE"
    return PolicyResult(
        outcome,
        amount,
        tenor,
        limit["indicative_rate_band"],
        round(champion_pd * policy["lgd"] * amount, 2),
        tuple(checks),
    )
