"""Structured logging setup using stdlib logging with JSON-ish formatting."""

import logging
import sys

from app.core.config import settings


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure and return the application root logger.

    Uses structured format: [timestamp] [level] [module] message
    """
    log_level = level or ("DEBUG" if settings.DEBUG else "INFO")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("sme_credit")
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return root_logger


logger = setup_logging()
