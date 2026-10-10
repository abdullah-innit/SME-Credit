"""Statement quality checks and privacy-safe reconciliation report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import median

from app.services.ingest.normalise import NormalizedRow


@dataclass(frozen=True)
class ValidationResult:
    unique_rows: list[NormalizedRow]
    report: dict


def _months(start: date, end: date) -> list[str]:
    result = []
    index = start.year * 12 + start.month - 1
    last = end.year * 12 + end.month - 1
    while index <= last:
        result.append(f"{index // 12:04d}-{index % 12 + 1:02d}")
        index += 1
    return result


def validate_statement(rows: list[NormalizedRow], row_errors: list[dict]) -> ValidationResult:
    seen: set[str] = set()
    unique = []
    duplicates = 0
    for row in rows:
        if row.row_hash in seen:
            duplicates += 1
        else:
            seen.add(row.row_hash)
            unique.append(row)
    if not unique:
        raise ValueError("Statement has no unique transactions")
    if unique[0].date > unique[-1].date:
        ordered = list(reversed(unique))
    else:
        ordered = sorted(unique, key=lambda row: row.date)
    mismatches = []
    for index, (previous, current) in enumerate(zip(ordered[:-1], ordered[1:], strict=True), 2):
        if previous.balance_after is None or current.balance_after is None:
            continue
        signed = current.amount_pkr if current.direction == "inflow" else -current.amount_pkr
        if abs(previous.balance_after + signed - current.balance_after) > 1:
            mismatches.append(index)
    start, end = min(row.date for row in unique), max(row.date for row in unique)
    expected = _months(start, end)
    observed = {f"{row.date.year:04d}-{row.date.month:02d}" for row in unique}
    missing = [month for month in expected if month not in observed]
    amounts = [row.amount_pkr for row in unique]
    threshold = max(1000000, 10 * median(amounts))
    outliers = [index for index, row in enumerate(unique, 2) if row.amount_pkr > threshold]
    balanced_pairs = sum(
        a.balance_after is not None and b.balance_after is not None
        for a, b in zip(ordered[:-1], ordered[1:], strict=True)
    )
    quality = max(
        0,
        100
        - min(40, len(mismatches) * 10)
        - min(20, len(missing) * 8)
        - min(15, len(row_errors) * 5)
        - min(10, len(outliers) * 2)
        - min(10, duplicates),
    )
    return ValidationResult(
        unique,
        {
            "data_quality_score": quality,
            "transaction_count": len(unique),
            "duplicate_rows": duplicates,
            "row_errors": row_errors,
            "balance_reconciliation": {
                "tested_pairs": balanced_pairs,
                "mismatches": len(mismatches),
                "mismatch_rows": mismatches,
                "passed": len(mismatches) == 0,
            },
            "coverage": {
                "from": start.isoformat(),
                "to": end.isoformat(),
                "calendar_days": (end - start).days + 1,
                "observed_months": len(observed),
                "expected_months": len(expected),
                "missing_months": missing,
            },
            "outliers": {"count": len(outliers), "rows": outliers, "threshold_pkr": int(threshold)},
        },
    )
