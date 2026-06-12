"""
app/main.py
============
FastAPI application entry point for Locate Anything Assistant.

Startup sequence
----------------
1. Initialise Loguru logging.
2. Load LocateAnything-3B via ModelService.load() — once, at startup.
3. Register API routers (/health, /detect, /chat).
4. Configure CORS middleware for the React frontend.

The model is intentionally loaded inside the lifespan context manager so
that it is guaranteed to be ready before Uvicorn starts accepting requests,
and properly released on shutdown.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, detect, health
from app.core.config import get_settings
from app.core.logging import _setup_logging, logger
from app.services.model_service import get_model_service


# ---------------------------------------------------------------------------
# Lifespan — model loads here, once, before any request is handled
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Everything before `yield` runs at startup;
    everything after runs at shutdown.
    """
    # Initialise structured logging first so all subsequent output is captured.
    _setup_logging()
    logger.info("=== Locate Anything Assistant starting up ===")

    settings = get_settings()
    logger.info(
        "Config | model={} | device={} | dtype={} | mode={}",
        settings.model_path,
        settings.device,
        settings.torch_dtype,
        settings.generation_mode,
    )

    # Load the model — this is the only place load() is called.
    svc = get_model_service()
    try:
        svc.load()
        logger.info("Model loaded successfully. Ready to serve requests.")
    except Exception as exc:
        # Log the error but do not crash the server.
        # The /health endpoint will report model_loaded=false,
        # and inference endpoints will return 503 until the model is available.
        logger.error("Model failed to load at startup: {}", exc)
        logger.warning(
            "Server is running WITHOUT a loaded model. "
            "Inference endpoints will return 503. "
            "Check MODEL_PATH and HF_TOKEN in your .env file."
        )

    yield  # <-- server is live and handling requests here

    # Shutdown: release model memory.
    logger.info("=== Locate Anything Assistant shutting down ===")
    try:
        svc.unload()
    except Exception as exc:
        logger.warning("Error during model unload: {}", exc)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Locate Anything Assistant",
        description=(
            "Production API for nvidia/LocateAnything-3B — "
            "a vision-language model for fast, precise visual grounding. "
            "Supports object detection, phrase grounding, GUI element grounding, "
            "scene text detection, and pointing tasks."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ----- CORS -----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----- Routers -----
    app.include_router(health.router, tags=["Health"])
    app.include_router(detect.router, tags=["Detection"])
    app.include_router(chat.router, tags=["Chat"])

    return app


# Create the application instance used by Uvicorn.
app = create_app()


# ---------------------------------------------------------------------------
# Entry point for `python -m app.main` (development convenience)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,           # Do NOT use reload with GPU models — causes double load
        workers=settings.workers,
        log_level=settings.log_level,
    )
