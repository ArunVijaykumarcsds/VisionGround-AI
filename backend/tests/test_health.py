"""
tests/test_health.py
=====================
Tests for GET /health endpoint.

Covers:
- 200 response when model is loaded
- Correct response schema fields
- model_loaded reflects singleton state
- memory_info present when model loaded
- device and model_path fields
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, PropertyMock


@pytest.mark.asyncio
async def test_health_returns_200(client):
    """GET /health always returns 200."""
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_schema(client):
    """Response contains all required fields."""
    response = await client.get("/health")
    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert "model_path" in data


@pytest.mark.asyncio
async def test_health_status_ok(client):
    """status field is always 'ok'."""
    response = await client.get("/health")
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_model_path(client):
    """model_path matches MODEL_PATH setting."""
    response = await client.get("/health")
    data = response.json()
    assert data["model_path"] == "nvidia/LocateAnything-3B"


@pytest.mark.asyncio
async def test_health_model_loaded_true_when_ready(client):
    """model_loaded is True when ModelService reports loaded."""
    with patch(
        "app.services.model_service.ModelService.is_loaded",
        new_callable=lambda: property(lambda self: True),
    ):
        response = await client.get("/health")
        assert response.json()["model_loaded"] is True


@pytest.mark.asyncio
async def test_health_model_loaded_false_when_not_ready(client):
    """model_loaded is False when model has not been loaded."""
    with patch(
        "app.services.model_service.ModelService.is_loaded",
        new_callable=lambda: property(lambda self: False),
    ):
        response = await client.get("/health")
        assert response.json()["model_loaded"] is False


@pytest.mark.asyncio
async def test_health_memory_info_present_when_loaded(client):
    """memory_info dict is included when model is loaded and memory logging enabled."""
    fake_mem = {
        "system_ram_total_gb": 32.0,
        "system_ram_used_gb": 10.0,
        "system_ram_available_gb": 22.0,
        "gpu_name": "NVIDIA RTX 3090",
        "gpu_vram_allocated_gb": 7.2,
        "gpu_vram_reserved_gb": 7.5,
        "gpu_vram_total_gb": 24.0,
    }
    with patch("app.services.model_service.ModelService.memory_info", return_value=fake_mem):
        with patch(
            "app.services.model_service.ModelService.is_loaded",
            new_callable=lambda: property(lambda self: True),
        ):
            response = await client.get("/health")
            data = response.json()
            assert data["memory_info"] is not None
            assert "system_ram_total_gb" in data["memory_info"]


@pytest.mark.asyncio
async def test_health_content_type_json(client):
    """Response Content-Type is application/json."""
    response = await client.get("/health")
    assert "application/json" in response.headers["content-type"]
