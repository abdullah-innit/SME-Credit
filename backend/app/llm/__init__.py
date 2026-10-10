"""LLM integration layer — safe, testable gateway for all LLM calls."""

from app.llm.fallbacks import get_deterministic_fallback
from app.llm.gateway import generate, get_health_info
from app.llm.grounding import check_numbers, extract_numbers
from app.llm.prompt_registry import (
    get_prompt_version,
    get_system_prompt,
    get_task_prompt,
    list_tasks,
)
from app.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    MockProvider,
    OpenAICompatibleProvider,
    get_provider,
)
from app.llm.sanitise import mask_api_key, sanitise_user_data
from app.llm.schemas import (
    TASK_SCHEMAS,
    AdvisorOutput,
    CreditMemoOutput,
    LLMError,
    LLMGroundingError,
    LLMParseError,
    LLMRateLimitError,
    LLMTimeoutError,
    StatementExtractionOutput,
)

__all__ = [
    "generate",
    "get_health_info",
    "get_deterministic_fallback",
    "check_numbers",
    "extract_numbers",
    "get_system_prompt",
    "get_task_prompt",
    "get_prompt_version",
    "list_tasks",
    "get_provider",
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "MockProvider",
    "sanitise_user_data",
    "mask_api_key",
    "TASK_SCHEMAS",
    "CreditMemoOutput",
    "StatementExtractionOutput",
    "AdvisorOutput",
    "LLMError",
    "LLMParseError",
    "LLMGroundingError",
    "LLMTimeoutError",
    "LLMRateLimitError",
]
