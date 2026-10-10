"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration – every value comes from .env or the environment."""

    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "SME Credit"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── LLM ──────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    OFFLINE_MODE: bool = True  # Default True so app works without API keys

    # ── Statement ingestion ─
    MAX_STATEMENT_BYTES: int = 10_000_000
    INGEST_DEBUG_KEEP_RAW: bool = False

    # ── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./sme_credit.db"

    # ── Auth ─────────────────────────────────────────────────────────
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # ── CORS ─────────────────────────────────────────────────────────
    FRONTEND_ORIGIN: str = "http://localhost:3000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton – import this everywhere
settings = Settings()
