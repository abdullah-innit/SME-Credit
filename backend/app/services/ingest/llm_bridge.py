"""Optional Prompt 10 LLM hooks; no network use until a provider is installed."""

from __future__ import annotations

import importlib
import json
import re
from typing import Any

from app.core.config import settings

ACCOUNT_RE = re.compile(r"\b(?:PK\d{2}[A-Z0-9]{16,26}|\d{8,24})\b", re.IGNORECASE)


def mask_accounts(value: str) -> str:
    return ACCOUNT_RE.sub("[ACCOUNT]", value)


def _provider() -> Any | None:
    if settings.OFFLINE_MODE or not settings.LLM_API_KEY:
        return None
    try:
        return importlib.import_module("app.llm.ingest")
    except ImportError:
        return None


def _json_object(value: Any) -> dict | None:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def propose_mapping(headers: list[str], sample: list[dict[str, Any]]) -> dict[str, str] | None:
    provider = _provider()
    if provider is None or not hasattr(provider, "propose_statement_mapping"):
        return None
    masked_rows = [
        {mask_accounts(str(k)): mask_accounts(str(v)) for k, v in row.items()} for row in sample
    ]
    try:
        return _json_object(
            provider.propose_statement_mapping(
                [mask_accounts(header) for header in headers], masked_rows
            )
        )
    except Exception:
        return None


def propose_categories(descriptions: list[str]) -> dict[str, dict] | None:
    provider = _provider()
    if provider is None or not hasattr(provider, "categorize_transactions"):
        return None
    masked = [mask_accounts(item) for item in descriptions]
    try:
        return _json_object(provider.categorize_transactions(masked))
    except Exception:
        return None
