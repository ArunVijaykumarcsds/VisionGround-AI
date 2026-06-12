"""
app/api/health.py
==================
GET /health  —  Liveness and readiness check endpoint.

Returns model load status, device info, and optional memory diagnostics.
Used by Docker health checks, load balancers, and monitoring systems.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.detection import HealthResponse
from app.services.model_service import get_model_service

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    description=(
        "Returns model load status, device, and memory diagnostics. "
        "A 200 response with model_loaded=true means the service is ready to serve requests."
    ),
)
async def health() -> HealthResponse:
    """
    Liveness + readiness probe.

    Returns status='ok' always (so Docker knows the process is alive).
    model_loaded indicates whether inference is available.
    """
    settings = get_settings()
    svc = get_model_service()

    mem_info = None
    if svc.is_loaded and settings.enable_memory_logging:
        try:
            mem_info = svc.memory_info()
        except Exception as exc:
            logger.warning("Could not retrieve memory info: {}", exc)

    logger.debug("GET /health | model_loaded={}", svc.is_loaded)

    return HealthResponse(
        status="ok",
        model_loaded=svc.is_loaded,
        device=settings.device,
        model_path=settings.model_path,
        memory_info=mem_info,
    )
