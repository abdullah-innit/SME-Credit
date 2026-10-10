"""Prompt-injection sanitisation and input hygiene utilities."""

from __future__ import annotations

import re

# Phrases that could hijack system instructions when embedded in user data
_CONTROL_PHRASES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
        r"you\s+are\s+now\s+(a|an|the)\s+",
        r"system\s*:\s*",
        r"<\s*/?\s*system\s*>",
        r"IMPORTANT\s*:\s*override",
        r"disregard\s+(your|all|the)\s+(rules?|instructions?|guidelines?)",
        r"forget\s+(your|all|the)\s+(rules?|instructions?|guidelines?)",
        r"new\s+instructions?\s*:",
        r"\[\s*INST\s*\]",
        r"<<\s*SYS\s*>>",
    ]
]


def sanitise_user_data(text: str) -> str:
    """Strip control phrases from user-supplied data (statements, CSV fields, etc.).

    This ensures uploaded content is treated as *data* not instructions.
    Returns the sanitised text.
    """
    if not text:
        return text
    cleaned = text
    for pattern in _CONTROL_PHRASES:
        cleaned = pattern.sub("[FILTERED]", cleaned)
    # Remove any null bytes or other control characters (except newlines/tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    return cleaned


def mask_api_key(key: str) -> str:
    """Mask an API key for safe logging: show first 4 and last 4 chars only."""
    if not key or len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"
