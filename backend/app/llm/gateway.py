"""LLM Gateway — safe, testable, and guarded single entry point for all LLM calls."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import enums
from app.core.config import settings
from app.db.models import AiCallLog
from app.db.session import SessionLocal
from app.llm.cache import (
    compute_cache_key,
    get_cached_response,
    rate_limiter,
    set_cached_response,
)
from app.llm.fallbacks import get_deterministic_fallback
from app.llm.grounding import check_numbers
from app.llm.prompt_registry import get_system_prompt, get_task_prompt
from app.llm.providers import BaseLLMProvider, get_provider
from app.llm.sanitise import sanitise_user_data
from app.llm.schemas import (
    TASK_SCHEMAS,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# In-memory tracking for health check
_HEALTH_STATE: dict[str, Any] = {
    "last_success_at": None,
    "last_call_at": None,
    "last_error": None,
    "last_error_at": None,
}


def get_health_info() -> dict[str, Any]:
    """Return health metrics for the LLM service layer."""
    provider_name = settings.LLM_PROVIDER
    mode = "offline" if settings.OFFLINE_MODE else "online"
    has_api_key = bool(settings.LLM_API_KEY.strip()) if not settings.OFFLINE_MODE else True

    status = "healthy"
    if not settings.OFFLINE_MODE and not has_api_key:
        status = "degraded (missing API key)"

    return {
        "status": status,
        "provider": provider_name,
        "model": settings.LLM_MODEL,
        "mode": mode,
        "has_api_key": bool(settings.LLM_API_KEY.strip()),
        "last_success_at": _HEALTH_STATE["last_success_at"],
        "last_call_at": _HEALTH_STATE["last_call_at"],
        "last_error": _HEALTH_STATE["last_error"],
        "rate_limit_rpm": settings.LLM_RATE_LIMIT_RPM,
    }


def _record_ai_call_log(
    *,
    application_id: int | None,
    actor_id: int | None,
    provider: str,
    model: str,
    task: str,
    status: enums.AiCallStatus,
    input_hash: str,
    output_hash: str,
    duration_ms: int,
    prompt_version: str,
    tokens: int = 0,
    cost_usd: float = 0.0,
    db: Session | None = None,
) -> None:
    """Safely persist AiCallLog without exposing sensitive tokens or keys."""
    try:
        # Enforce exact 64-char lowercase hex constraints
        valid_in_hash = (
            input_hash if len(input_hash) == 64 else hashlib.sha256(input_hash.encode()).hexdigest()
        )
        valid_out_hash = (
            output_hash
            if len(output_hash) == 64
            else hashlib.sha256(output_hash.encode()).hexdigest()
        )

        log_entry = AiCallLog(
            application_id=application_id,
            actor=actor_id,
            provider=provider,
            model=model,
            task=task,
            status=status,
            input_hash=valid_in_hash,
            output_hash=valid_out_hash,
            duration_ms=max(0, duration_ms),
            prompt_version=prompt_version,
            tokens=max(0, tokens),
            cost_usd=max(0.0, float(cost_usd)),
        )

        if db is not None:
            db.add(log_entry)
            db.commit()
        else:
            with SessionLocal() as session:
                session.add(log_entry)
                session.commit()
    except Exception as exc:
        logger.warning("Failed to record AiCallLog: %s", exc)


def _sanitise_facts(facts: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitise string values in facts bundle to prevent prompt injection."""
    if not facts:
        return {}
    cleaned: dict[str, Any] = {}
    for k, v in facts.items():
        if isinstance(v, str):
            cleaned[k] = sanitise_user_data(v)
        elif isinstance(v, dict):
            cleaned[k] = _sanitise_facts(v)
        else:
            cleaned[k] = v
    return cleaned


def _format_facts_for_prompt(facts: dict[str, Any]) -> str:
    """Format facts into readable string for prompt injection."""
    lines: list[str] = []
    for k, v in sorted(facts.items()):
        if isinstance(v, dict):
            lines.append(f"- {k}:")
            for sub_k, sub_v in sorted(v.items()):
                lines.append(f"    {sub_k}: {sub_v}")
        else:
            lines.append(f"- {k} [{k}]: {v}")
    return "\n".join(lines)


def _extract_json_block(text: str) -> str:
    """Extract JSON content from markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def generate(
    task: str,
    *,
    facts: dict[str, Any] | None = None,
    context: str = "",
    application_id: int | None = None,
    actor_id: int | None = None,
    provider: BaseLLMProvider | None = None,
    model: str | None = None,
    db: Session | None = None,
    bypass_cache: bool = False,
) -> BaseModel:
    """Execute LLM task with full guardrails, validation, caching, and fallback."""
    start_time = time.time()
    _HEALTH_STATE["last_call_at"] = datetime.now(UTC)

    if task not in TASK_SCHEMAS:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {list(TASK_SCHEMAS.keys())}")

    schema_cls = TASK_SCHEMAS[task]
    sanitised_context = sanitise_user_data(context)
    clean_facts = _sanitise_facts(facts)
    facts_str = _format_facts_for_prompt(clean_facts)

    prompt_version, rendered_prompt = get_task_prompt(
        task, facts=facts_str, context=sanitised_context
    )
    system_prompt = get_system_prompt()

    active_provider = provider or get_provider()
    active_model = model or settings.LLM_MODEL
    input_hash = compute_cache_key(
        task, prompt_version, system_prompt, rendered_prompt, clean_facts
    )

    # 1. Deterministic Fallback on OFFLINE_MODE
    if (
        settings.OFFLINE_MODE
        and not isinstance(active_provider, BaseLLMProvider)
        and provider is None
    ):
        pass  # Handled below

    if settings.OFFLINE_MODE and (provider is None or provider.name != "mock"):
        fallback_obj = get_deterministic_fallback(
            task, facts=clean_facts, context=sanitised_context
        )
        duration_ms = int((time.time() - start_time) * 1000)
        output_json = fallback_obj.model_dump_json()
        out_hash = hashlib.sha256(output_json.encode()).hexdigest()

        _record_ai_call_log(
            application_id=application_id,
            actor_id=actor_id,
            provider=active_provider.name,
            model=active_model,
            task=task,
            status=enums.AiCallStatus.OFFLINE,
            input_hash=input_hash,
            output_hash=out_hash,
            duration_ms=duration_ms,
            prompt_version=prompt_version,
            tokens=0,
            cost_usd=0.0,
            db=db,
        )
        _HEALTH_STATE["last_success_at"] = datetime.now(UTC)
        return fallback_obj

    # 2. Check Cache
    if not bypass_cache:
        cached = get_cached_response(input_hash)
        if cached:
            try:
                validated = schema_cls.model_validate_json(cached["response_json"])
                duration_ms = int((time.time() - start_time) * 1000)
                out_hash = hashlib.sha256(cached["response_json"].encode()).hexdigest()

                _record_ai_call_log(
                    application_id=application_id,
                    actor_id=actor_id,
                    provider=active_provider.name,
                    model=active_model,
                    task=task,
                    status=enums.AiCallStatus.SUCCESS,
                    input_hash=input_hash,
                    output_hash=out_hash,
                    duration_ms=duration_ms,
                    prompt_version=prompt_version,
                    tokens=cached.get("prompt_tokens", 0) + cached.get("completion_tokens", 0),
                    cost_usd=cached.get("cost_usd", 0.0),
                    db=db,
                )
                _HEALTH_STATE["last_success_at"] = datetime.now(UTC)
                return validated
            except Exception:
                pass  # Fall through to re-generate if cache corrupted

    # 3. Rate Limiting
    rate_limiter.acquire()

    # 4. Generate with Provider (with Timeout, Backoff & Error Handling)
    response = None
    last_err: Exception | None = None
    retries = settings.LLM_MAX_RETRIES

    for attempt in range(retries + 1):
        try:
            response = active_provider.generate(
                rendered_prompt,
                system=system_prompt,
                model=active_model,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            )
            break
        except (LLMTimeoutError, LLMRateLimitError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
            else:
                break
        except Exception as exc:
            last_err = exc
            break

    # If provider failed (timeout, rate limit, network): fall back safely
    if response is None:
        _HEALTH_STATE["last_error"] = str(last_err)
        _HEALTH_STATE["last_error_at"] = datetime.now(UTC)
        duration_ms = int((time.time() - start_time) * 1000)
        fallback_obj = get_deterministic_fallback(
            task, facts=clean_facts, context=sanitised_context
        )
        out_hash = hashlib.sha256(fallback_obj.model_dump_json().encode()).hexdigest()

        _record_ai_call_log(
            application_id=application_id,
            actor_id=actor_id,
            provider=active_provider.name,
            model=active_model,
            task=task,
            status=enums.AiCallStatus.FAILED,
            input_hash=input_hash,
            output_hash=out_hash,
            duration_ms=duration_ms,
            prompt_version=prompt_version,
            tokens=0,
            cost_usd=0.0,
            db=db,
        )
        return fallback_obj

    # 5. Parse and Validate Schema (with Automatic Repair Retry)
    content_raw = _extract_json_block(response.content)
    parsed_obj: BaseModel | None = None

    try:
        parsed_dict = json.loads(content_raw)
        parsed_obj = schema_cls.model_validate(parsed_dict)
    except Exception as parse_exc:
        # Automatic Repair Retry: 1 attempt to repair invalid JSON
        logger.info("Task '%s' initial parse failed (%s); triggering repair retry", task, parse_exc)
        repair_prompt = (
            f"Your previous response failed JSON validation against the schema: {parse_exc}\n"
            f"Here is your previous response:\n{response.content}\n\n"
            f"Please output ONLY valid, strictly conforming JSON matching this schema:\n"
            f"{schema_cls.model_json_schema()}"
        )
        try:
            repair_res = active_provider.generate(
                repair_prompt,
                system=system_prompt,
                model=active_model,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            )
            repaired_raw = _extract_json_block(repair_res.content)
            repaired_dict = json.loads(repaired_raw)
            parsed_obj = schema_cls.model_validate(repaired_dict)
            response.prompt_tokens += repair_res.prompt_tokens
            response.completion_tokens += repair_res.completion_tokens
            response.cost_usd += repair_res.cost_usd
        except Exception:
            parsed_obj = None

    # If parsing and repair both failed: fall back
    if parsed_obj is None:
        duration_ms = int((time.time() - start_time) * 1000)
        fallback_obj = get_deterministic_fallback(
            task, facts=clean_facts, context=sanitised_context
        )
        out_hash = hashlib.sha256(fallback_obj.model_dump_json().encode()).hexdigest()
        _record_ai_call_log(
            application_id=application_id,
            actor_id=actor_id,
            provider=active_provider.name,
            model=active_model,
            task=task,
            status=enums.AiCallStatus.FAILED,
            input_hash=input_hash,
            output_hash=out_hash,
            duration_ms=duration_ms,
            prompt_version=prompt_version,
            tokens=response.prompt_tokens + response.completion_tokens,
            cost_usd=response.cost_usd,
            db=db,
        )
        return fallback_obj

    # 6. Number Grounding Checker
    violations = check_numbers(parsed_obj.model_dump_json(), clean_facts)
    if violations:
        logger.warning(
            "Task '%s' output contained ungrounded numbers: %s; triggering regeneration retry",
            task,
            violations,
        )
        grounding_retry_prompt = (
            "Your output contained numeric values that do not exist in the facts "
            f"bundle or whitelisted arithmetic: {violations}\n"
            "CRITICAL RULE: YOU NEVER INVENT NUMBERS. Every number must come from "
            f"the facts bundle:\n{facts_str}\n\n"
            "Regenerate the response adhering strictly to the facts."
        )
        try:
            regen_res = active_provider.generate(
                grounding_retry_prompt,
                system=system_prompt,
                model=active_model,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            )
            regen_raw = _extract_json_block(regen_res.content)
            regen_obj = schema_cls.model_validate(json.loads(regen_raw))
            second_violations = check_numbers(regen_obj.model_dump_json(), clean_facts)
            if not second_violations:
                parsed_obj = regen_obj
                response.prompt_tokens += regen_res.prompt_tokens
                response.completion_tokens += regen_res.completion_tokens
                response.cost_usd += regen_res.cost_usd
            else:
                # Still ungrounded: fall back to deterministic safe fallback
                parsed_obj = get_deterministic_fallback(
                    task, facts=clean_facts, context=sanitised_context
                )
        except Exception:
            parsed_obj = get_deterministic_fallback(
                task, facts=clean_facts, context=sanitised_context
            )

    # 7. Record AiCallLog and Cache Result
    duration_ms = int((time.time() - start_time) * 1000)
    final_json = parsed_obj.model_dump_json()
    out_hash = hashlib.sha256(final_json.encode()).hexdigest()

    set_cached_response(
        input_hash,
        final_json,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        cost_usd=response.cost_usd,
    )

    _record_ai_call_log(
        application_id=application_id,
        actor_id=actor_id,
        provider=active_provider.name,
        model=active_model,
        task=task,
        status=enums.AiCallStatus.SUCCESS,
        input_hash=input_hash,
        output_hash=out_hash,
        duration_ms=duration_ms,
        prompt_version=prompt_version,
        tokens=response.prompt_tokens + response.completion_tokens,
        cost_usd=response.cost_usd,
        db=db,
    )

    _HEALTH_STATE["last_success_at"] = datetime.now(UTC)
    return parsed_obj
