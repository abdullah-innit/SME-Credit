"""Evaluate action-plan coverage on the untouched business-disjoint test split."""

from __future__ import annotations

import json
from time import perf_counter

from sqlalchemy import select

from app.db.models import Application
from app.db.session import SessionLocal
from app.ml.counterfactual import ACTION_FEATURE, ScenarioChanges, find_path_to_approval, what_if
from app.ml.registry import FEATURES
from app.ml.scorecard import ARTIFACTS
from app.services.pipeline import warm_models


def main() -> None:
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    test_ids = set(metadata["split_business_ids"]["test"])
    warm_models()
    with SessionLocal() as session:
        applications = session.scalars(
            select(Application)
            .where(Application.business_id.in_(test_ids))
            .order_by(Application.id)
        ).all()
        total = found = verified = 0
        start = perf_counter()
        for index, app in enumerate(applications, 1):
            result = find_path_to_approval(app.id, session=session, lender_view=True)
            if result["original_decision"] != "APPROVE":
                total += 1
                if result["plans"]:
                    found += 1
                for plan in result["plans"]:
                    assert all(FEATURES[ACTION_FEATURE[key]].actionable for key in plan["changes"])
                    preview = what_if(app.id, ScenarioChanges(**plan["changes"]), session=session)
                    assert preview["what_if"]["decision"] == "APPROVE"
                    assert preview["what_if"]["pd"] <= 0.095
                    verified += 1
            if index % 50 == 0:
                session.rollback()
                print(f"Evaluated {index}/{len(applications)}", flush=True)
        session.rollback()
    print(
        json.dumps(
            {
                "test_applications": len(applications),
                "referred_or_declined": total,
                "plan_found": found,
                "coverage_percent": round(100 * found / total, 1) if total else 0,
                "plans_reapplied_and_approved": verified,
                "elapsed_seconds": round(perf_counter() - start, 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
