"""Normalise statement rows and categorise with rules before optional LLM batching."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from openpyxl.utils.datetime import from_excel

from app.services.ingest.llm_bridge import mask_accounts, propose_categories
from app.services.ingest.parsers import ParseError

CATEGORIES = frozenset(
    {
        "sales",
        "supplier_payment",
        "payroll",
        "rent",
        "utilities",
        "loan_repayment",
        "owner_withdrawal",
        "owner_injection",
        "tax",
        "other",
    }
)
RULES = (
    ("owner_withdrawal", ("owner withdrawal", "drawings", "personal withdrawal")),
    ("owner_injection", ("owner injection", "capital injection", "owner deposit")),
    ("loan_repayment", ("loan repayment", "loan payment", "emi", "finance instalment")),
    ("supplier_payment", ("supplier", "vendor", "inventory", "purchase")),
    ("payroll", ("payroll", "salary", "salaries", "wages")),
    ("rent", ("rent", "lease payment")),
    ("utilities", ("utility", "electricity", "gas bill", "water bill", "internet bill")),
    ("tax", ("tax", "fbr", "sales tax", "withholding")),
    (
        "sales",
        ("sale", "invoice", "pos settlement", "pos_settlement", "customer", "digital transfer in"),
    ),
)
_LLM_CACHE: dict[str, tuple[str, float]] = {}


@dataclass(frozen=True)
class NormalizedRow:
    date: date
    amount_pkr: int
    direction: str
    channel: str
    category: str
    counterparty_hash: str
    balance_after: int | None
    row_hash: str
    category_confidence: float
    category_source: str


def _date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        return from_excel(value).date()
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ParseError("Unrecognised transaction date")


def _money(value: Any) -> int:
    if value is None or str(value).strip() in {"", "-"}:
        return 0
    text = str(value).strip().upper().replace("PKR", "").replace(",", "").replace(" ", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if text.endswith("DR"):
        negative = True
        text = text[:-2]
    elif text.endswith("CR"):
        text = text[:-2]
    try:
        amount = float(text)
    except ValueError as exc:
        raise ParseError("Invalid transaction amount") from exc
    if not amount.is_integer():
        raise ParseError("Transaction amount must be whole PKR")
    return -int(amount) if negative else int(amount)


def _channel(value: Any, description: str) -> str:
    text = f"{value or ''} {description}".lower().replace("_", " ")
    if "wallet" in text or "easypaisa" in text or "jazzcash" in text:
        return "wallet"
    if "cash" in text:
        return "cash_deposit"
    if "instant" in text or "raast" in text or "ibft" in text:
        return "instant_transfer"
    return "bank"


def _rule_category(description: str, category_hint: str) -> tuple[str, float]:
    text = f"{category_hint} {description}".lower().replace("_", " ")
    for category, patterns in RULES:
        if any(pattern in text for pattern in patterns):
            return category, 0.98 if category_hint else 0.9
    return "other", 0.4


def _digest(parts: tuple[Any, ...]) -> str:
    return hashlib.sha256("|".join(str(item) for item in parts).encode("utf-8")).hexdigest()


def normalise_rows(
    raw_rows: list[dict[str, Any]],
    mapping: dict[str, str],
    *,
    business_id: int,
    offline: bool = True,
) -> tuple[list[NormalizedRow], list[dict]]:
    pending: list[dict] = []
    errors: list[dict] = []
    for number, raw in enumerate(raw_rows, 2):
        try:
            day = _date(raw.get(mapping["date"]))
            description = mask_accounts(str(raw.get(mapping.get("description", ""), "") or ""))
            category_hint = mask_accounts(str(raw.get(mapping.get("category", ""), "") or ""))
            if "amount" in mapping:
                signed = _money(raw.get(mapping["amount"]))
                direction_hint = str(raw.get(mapping.get("direction", ""), "") or "").lower()
                if direction_hint in {"debit", "dr", "outflow"}:
                    signed = -abs(signed)
                elif direction_hint in {"credit", "cr", "inflow"}:
                    signed = abs(signed)
            else:
                inflow = _money(raw.get(mapping["inflow"]))
                outflow = _money(raw.get(mapping["outflow"]))
                if inflow < 0 or outflow < 0 or (inflow > 0 and outflow > 0):
                    raise ParseError("Inflow and outflow must be separate nonnegative amounts")
                signed = inflow - outflow
            if signed == 0:
                raise ParseError("Transaction has no amount")
            balance_value = raw.get(mapping.get("balance", ""))
            balance = _money(balance_value) if balance_value not in (None, "") else None
            if balance is not None and balance < 0:
                raise ParseError("Running balance cannot be negative")
            direction = "inflow" if signed > 0 else "outflow"
            channel = _channel(raw.get(mapping.get("channel", "")), description)
            counterparty = str(raw.get(mapping.get("counterparty", ""), "") or "")
            if re.fullmatch(r"[0-9a-f]{64}", counterparty):
                counterparty_hash = counterparty
            else:
                counterparty_hash = _digest(
                    (business_id, mask_accounts(counterparty or description).lower())
                )
            category, confidence = _rule_category(description, category_hint)
            pending.append(
                {
                    "date": day,
                    "amount_pkr": abs(signed),
                    "direction": direction,
                    "channel": channel,
                    "category": category,
                    "confidence": confidence,
                    "description": description,
                    "counterparty_hash": counterparty_hash,
                    "balance_after": balance,
                }
            )
        except ParseError as exc:
            errors.append({"row": number, "code": "INVALID_ROW", "message": str(exc)})
    if not pending:
        raise ParseError("No valid transactions; check date and amount columns")
    ambiguous = sorted(
        {
            row["description"]
            for row in pending
            if row["category"] == "other" and row["description"] not in _LLM_CACHE
        }
    )
    if ambiguous and not offline:
        for start in range(0, len(ambiguous), 25):
            batch = ambiguous[start : start + 25]
            proposals = propose_categories(batch) or {}
            for text in batch:
                choice = proposals.get(text)
                if isinstance(choice, dict) and choice.get("category") in CATEGORIES:
                    try:
                        confidence = float(choice.get("confidence", 0))
                    except (TypeError, ValueError):
                        continue
                    if 0.5 <= confidence <= 1:
                        _LLM_CACHE[text] = (choice["category"], confidence)
    result = []
    for row in pending:
        source = "rule"
        if row["category"] == "other" and row["description"] in _LLM_CACHE:
            row["category"], row["confidence"] = _LLM_CACHE[row["description"]]
            source = "llm"
        fingerprint = _digest(
            (
                row["date"],
                row["amount_pkr"],
                row["direction"],
                row["channel"],
                row["balance_after"]
                if row["balance_after"] is not None
                else row["counterparty_hash"],
            )
        )
        result.append(
            NormalizedRow(
                row["date"],
                row["amount_pkr"],
                row["direction"],
                row["channel"],
                row["category"],
                row["counterparty_hash"],
                row["balance_after"],
                fingerprint,
                row["confidence"],
                source,
            )
        )
    return result, errors
