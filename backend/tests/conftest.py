"""
tests/conftest.py
==================
Shared pytest fixtures used across all test modules.

The model is NEVER loaded in tests — all inference calls are mocked
via the ModelService.predict() method. This keeps the test suite fast,
GPU-free, and runnable in CI.

Fixtures provided
-----------------
app         : FastAPI application instance
client      : httpx AsyncClient bound to the app
fake_image  : a tiny 64x64 JPEG bytes object for upload tests
model_svc   : the ModelService singleton (not loaded)
mock_predict: patches ModelService.predict() to return a deterministic result
"""

from __future__ import annotations

import io
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image


# ---------------------------------------------------------------------------
# Application fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """
    Return the FastAPI app with model loading disabled.

    We patch ModelService.load() to a no-op so the lifespan hook
    does not attempt to download / load the real model.
    """
    with patch("app.services.model_service.ModelService.load", return_value=None):
        with patch("app.services.model_service.ModelService.is_loaded",
                   new_callable=lambda: property(lambda self: True)):
            from app.main import create_app
            return create_app()


# ---------------------------------------------------------------------------
# Async HTTP client fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTPX client wired to the FastAPI test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Image fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fake_image_bytes() -> bytes:
    """Return minimal valid JPEG bytes (64x64 RGB)."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def fake_image_file(fake_image_bytes):
    """Return a file-like tuple suitable for httpx multipart upload."""
    return ("image.jpg", io.BytesIO(fake_image_bytes), "image/jpeg")


# ---------------------------------------------------------------------------
# Mock predict fixture
# ---------------------------------------------------------------------------

MOCK_BOX_ANSWER = "<box><100><200><400><600></box><box><500><100><800><350></box>"
MOCK_POINT_ANSWER = "<box><350><480></box>"

@pytest.fixture()
def mock_predict_boxes():
    """
    Patch ModelService.predict() to return two deterministic bounding boxes.
    """
    result = {
        "answer": MOCK_BOX_ANSWER,
        "inference_time_ms": 123.4,
        "generation_mode_used": "hybrid",
    }
    with patch(
        "app.services.model_service.ModelService.predict",
        return_value=result,
    ) as mock:
        yield mock


@pytest.fixture()
def mock_predict_points():
    """
    Patch ModelService.predict() to return a single point.
    """
    result = {
        "answer": MOCK_POINT_ANSWER,
        "inference_time_ms": 98.0,
        "generation_mode_used": "fast",
    }
    with patch(
        "app.services.model_service.ModelService.predict",
        return_value=result,
    ) as mock:
        yield mock


@pytest.fixture()
def mock_predict_empty():
    """
    Patch ModelService.predict() to return an answer with no boxes/points.
    """
    result = {
        "answer": "I could not find any matching objects in this image.",
        "inference_time_ms": 87.0,
        "generation_mode_used": "hybrid",
    }
    with patch(
        "app.services.model_service.ModelService.predict",
        return_value=result,
    ) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Model service fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def model_svc():
    """Return the ModelService singleton (unloaded)."""
    from app.services.model_service import get_model_service
    return get_model_service()
