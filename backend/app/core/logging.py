"""
app/core/logging.py
====================
Centralised logging configuration using Loguru.

Loguru replaces the standard library's logging module with a far simpler API
that supports structured output, coloured console output, and async safety.

Usage anywhere in the codebase:
    from app.core.logging import logger
    logger.info("Ready")
    logger.debug("detail={}", some_var)
    logger.exception("Unhandled error")
"""

from __future__ import annotations

import sys
import logging

from loguru import logger

from app.core.config import get_settings


def _setup_logging() -> None:
    """
    Configure Loguru sink(s) based on application settings.

    Called once at startup from app.main.
    """
    settings = get_settings()

    # Remove the default Loguru stderr sink so we can reconfigure it.
    logger.remove()

    # Re-add stderr sink with structured format and chosen log level.
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Optionally add a file sink for persistent logs.
    logger.add(
        "logs/locate_anything_{time:YYYY-MM-DD}.log",
        format=log_format,
        level=settings.log_level.upper(),
        rotation="100 MB",
        retention="7 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,           # async-safe write via internal queue
    )

    # Intercept standard library logging so third-party libraries
    # (uvicorn, fastapi, transformers) emit through Loguru.
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno  # type: ignore[assignment]

            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back  # type: ignore[assignment]
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.access", "fastapi", "transformers"):
        std_logger = logging.getLogger(name)
        std_logger.handlers = [_InterceptHandler()]
        std_logger.propagate = False

    logger.info(
        "Logging initialised | level={} | device={}",
        settings.log_level.upper(),
        settings.device,
    )


# Export the configured logger for import throughout the application.
__all__ = ["logger", "_setup_logging"]
