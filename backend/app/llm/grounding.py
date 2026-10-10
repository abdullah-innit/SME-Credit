"""Number-grounding checker for LLM outputs.

Verifies that every number in a generated response is either:
1. Present in the facts bundle (exact match), or
2. A result of a whitelisted arithmetic operation on bundle values.

If a violation is found, the gateway triggers regeneration or falls back.
"""

from __future__ import annotations

import re
from typing import Any

# Numbers we always allow (percentages 0-100, counts 0-10, years)
_ALWAYS_ALLOWED = set(range(0, 101)) | set(range(2000, 2100))


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a string."""
    # Strip unicode escape sequences like \u0627 so hex digits aren't treated as numbers
    clean_text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)
    # Match integers and decimals, including negative and comma-separated
    raw = re.findall(r"-?[\d,]+\.?\d*", clean_text)
    result: list[float] = []
    for r in raw:
        try:
            cleaned = r.replace(",", "").strip()
            if cleaned and cleaned not in {"-", "."}:
                result.append(float(cleaned))
        except ValueError:
            continue
    return result


def _collect_values(obj: Any, values: list[float]) -> None:
    """Recursively collect all numeric values from arbitrary nested structures."""
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        values.append(float(obj))
    elif isinstance(obj, str):
        for n in extract_numbers(obj):
            values.append(n)
    elif isinstance(obj, dict):
        for val in obj.values():
            _collect_values(val, values)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            _collect_values(item, values)


def _build_allowed_set(
    facts: dict[str, Any],
) -> set[float]:
    """Build the set of allowed numbers from the facts bundle.

    Includes raw values and basic arithmetic combinations
    (sum, difference, product, ratio) of pairs.
    """
    allowed: set[float] = set()
    values: list[float] = []

    _collect_values(facts, values)

    allowed.update(values)
    for val in values:
        allowed.add(round(val, 1))
        allowed.add(round(val, 2))
        allowed.add(round(val, 3))
        allowed.add(round(val, 4))
        allowed.add(-val)
        allowed.add(abs(val))

    # Whitelisted operations on pairs
    for i, a in enumerate(values):
        for b in values[i:]:
            allowed.add(a + b)
            allowed.add(abs(a - b))
            if b != 0:
                allowed.add(round(a / b, 2))
                allowed.add(round(a / b, 3))
                allowed.add(round(a / b, 4))
            if a != 0:
                allowed.add(round(b / a, 2))
                allowed.add(round(b / a, 3))
                allowed.add(round(b / a, 4))
            allowed.add(a * b)

    # Always allowed universal numbers
    allowed.update(float(x) for x in _ALWAYS_ALLOWED)

    return allowed


def check_numbers(
    text: str,
    facts: dict[str, Any],
) -> list[float]:
    """Check that all numbers in text are grounded in the facts bundle.

    Returns a list of ungrounded numbers. Empty means all numbers are grounded.
    """
    numbers = extract_numbers(text)
    if not numbers:
        return []

    allowed = _build_allowed_set(facts)
    violations: list[float] = []

    for n in numbers:
        if n in allowed:
            continue
        # Check integer form too
        if n == int(n) and float(int(n)) in allowed:
            continue
        # Check standard rounded forms
        if (
            round(n, 1) in allowed
            or round(n, 2) in allowed
            or round(n, 3) in allowed
            or round(n, 4) in allowed
        ):
            continue
        violations.append(n)

    return violations
