"""Versioned prompt registry — loads prompt files and tracks versions."""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Cache: task_name -> (version_hash, system_text, task_text)
_cache: dict[str, tuple[str, str, str]] = {}


def _load(filename: str) -> str:
    """Load a prompt file and return its content."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _version_hash(text: str) -> str:
    """SHA-256 of the prompt text, truncated to 12 hex chars."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def get_system_prompt() -> str:
    """Return the shared system prompt."""
    return _load("system.txt")


def get_task_prompt(
    task: str,
    *,
    facts: str = "",
    context: str = "",
) -> tuple[str, str]:
    """Return (version_hash, rendered_prompt) for a task.

    The prompt template is loaded from ``prompts/<task>.txt`` and rendered
    with ``{facts}`` and ``{context}`` placeholders.
    """
    raw = _load(f"{task}.txt")
    rendered = raw.replace("{facts}", facts).replace("{context}", context)
    version = _version_hash(raw)
    return version, rendered


def get_prompt_version(task: str) -> str:
    """Return the version hash for a task prompt (without rendering)."""
    raw = _load(f"{task}.txt")
    return _version_hash(raw)


def list_tasks() -> list[str]:
    """List all available task names (prompt files without extension)."""
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.txt") if p.stem != "system")
