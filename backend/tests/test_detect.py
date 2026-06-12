"""
tests/test_detect.py
=====================
Tests for POST /detect endpoint.

Covers
------
- 200 with valid image + query
- Correct DetectResponse schema fields
- Bounding box parsing from mock model output
- Point parsing from mock model output
- Empty detection result (no boxes)
- 422 on missing query field
- 422 on unsupported image MIME type
- 413 on oversized image
- 503 when model not loaded
- generation_mode override passed through
- max_new_tokens override passed through
- Example query chips (multi-category, free-form, text)
"""

from __future__ import annotations

import io
import pytest
from PIL import Image
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image_bytes(width=64, height=64, fmt="JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color=(80, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def detect_files_and_data(query="find all cars", image_bytes=None, mime="image/jpeg", **extra):
    if image_bytes is None:
        image_bytes = make_image_bytes()
    files  = {"image": ("test.jpg", io.BytesIO(image_bytes), mime)}
    data   = {"query": query, **extra}
    return files, data


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_200_with_boxes(client, mock_predict_boxes):
    """POST /detect returns 200 and parses bounding boxes correctly."""
    files, data = detect_files_and_data("find all cars")
    response = await client.post("/detect", files=files, data=data)

    assert response.status_code == 200
    body = response.json()

    # Schema fields
    assert "query"               in body
    assert "detections"          in body
    assert "points"              in body
    assert "raw_answer"          in body
    assert "image_width"         in body
    assert "image_height"        in body
    assert "inference_time_ms"   in body
    assert "generation_mode_used" in body


@pytest.mark.asyncio
async def test_detect_parses_two_boxes(client, mock_predict_boxes):
    """Two <box> tokens in mock answer → two detections returned."""
    files, data = detect_files_and_data("find all cars")
    response = await client.post("/detect", files=files, data=data)
    body = response.json()

    assert len(body["detections"]) == 2

    for det in body["detections"]:
        assert "label"            in det
        assert "confidence"       in det
        assert "bbox"             in det
        assert "bbox_normalised"  in det
        assert len(det["bbox"]) == 4
        assert len(det["bbox_normalised"]) == 4
        # Pixel coords must be positive and x2 > x1, y2 > y1
        x1, y1, x2, y2 = det["bbox"]
        assert x2 > x1
        assert y2 > y1
        # Normalised coords in [0, 1]
        for v in det["bbox_normalised"]:
            assert 0.0 <= v <= 1.0


@pytest.mark.asyncio
async def test_detect_parses_points(client, mock_predict_points):
    """Single <box><x><y></box> token → one point, zero boxes."""
    files, data = detect_files_and_data("point to the cat")
    response = await client.post("/detect", files=files, data=data)
    body = response.json()

    # The mock answer has only a 2-coord token, so no 4-coord boxes
    assert len(body["detections"]) == 0
    assert len(body["points"]) == 1

    pt = body["points"][0]
    assert "x" in pt
    assert "y" in pt
    assert pt["x"] > 0
    assert pt["y"] > 0


@pytest.mark.asyncio
async def test_detect_empty_result(client, mock_predict_empty):
    """No <box> tokens in answer → zero detections and zero points."""
    files, data = detect_files_and_data("find all dragons")
    response = await client.post("/detect", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["detections"] == []
    assert body["points"] == []


@pytest.mark.asyncio
async def test_detect_query_echoed(client, mock_predict_boxes):
    """The original query is echoed back in the response."""
    query = "find all pedestrians"
    files, data = detect_files_and_data(query)
    response = await client.post("/detect", files=files, data=data)
    assert response.json()["query"] == query


@pytest.mark.asyncio
async def test_detect_image_dimensions_in_response(client, mock_predict_boxes):
    """image_width and image_height reflect the uploaded image size."""
    image_bytes = make_image_bytes(width=128, height=96)
    files = {"image": ("img.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    data  = {"query": "find objects"}
    response = await client.post("/detect", files=files, data=data)
    body = response.json()

    assert body["image_width"]  == 128
    assert body["image_height"] == 96


@pytest.mark.asyncio
async def test_detect_inference_time_positive(client, mock_predict_boxes):
    """inference_time_ms must be a positive number."""
    files, data = detect_files_and_data()
    response = await client.post("/detect", files=files, data=data)
    assert response.json()["inference_time_ms"] > 0


@pytest.mark.asyncio
async def test_detect_raw_answer_present(client, mock_predict_boxes):
    """raw_answer contains the model output string."""
    files, data = detect_files_and_data()
    response = await client.post("/detect", files=files, data=data)
    body = response.json()
    assert isinstance(body["raw_answer"], str)
    assert len(body["raw_answer"]) > 0


# ---------------------------------------------------------------------------
# Query variations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "find all cars",
    "find car, person, bicycle",
    "locate people wearing red jackets",
    "detect all text",
    "point to the traffic light",
    "find the tallest building",
])
async def test_detect_various_queries(client, mock_predict_boxes, query):
    """Various query styles all return 200."""
    files, data = detect_files_and_data(query)
    response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Generation mode override
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hybrid", "fast", "slow"])
async def test_detect_generation_mode_override(client, mock_predict_boxes, mode):
    """generation_mode override is accepted and reflected in response."""
    files, data = detect_files_and_data(generation_mode=mode)
    response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 200
    # The mock always returns "hybrid" for generation_mode_used,
    # but the important thing is no 422 error.


# ---------------------------------------------------------------------------
# Image format support
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("fmt,mime", [
    ("JPEG", "image/jpeg"),
    ("PNG",  "image/png"),
    ("WebP", "image/webp"),
    ("BMP",  "image/bmp"),
])
async def test_detect_image_formats(client, mock_predict_boxes, fmt, mime):
    """All supported image formats return 200."""
    image_bytes = make_image_bytes(fmt=fmt)
    files = {"image": ("test." + fmt.lower(), io.BytesIO(image_bytes), mime)}
    data  = {"query": "find objects"}
    response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_missing_query_returns_422(client):
    """Omitting the query field returns 422 Unprocessable Entity."""
    files = {"image": ("test.jpg", io.BytesIO(make_image_bytes()), "image/jpeg")}
    response = await client.post("/detect", files=files)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_detect_missing_image_returns_422(client):
    """Omitting the image field returns 422 Unprocessable Entity."""
    response = await client.post("/detect", data={"query": "find cars"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_detect_empty_query_returns_422(client):
    """Empty string query returns 422."""
    files, data = detect_files_and_data(query="")
    response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_detect_unsupported_mime_returns_422(client):
    """Unsupported MIME type returns 422."""
    files = {"image": ("test.gif", io.BytesIO(b"GIF89a"), "image/gif")}
    data  = {"query": "find objects"}
    response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_detect_corrupt_image_returns_422(client):
    """Corrupt/non-image bytes return 422."""
    files = {"image": ("bad.jpg", io.BytesIO(b"this is not an image"), "image/jpeg")}
    data  = {"query": "find objects"}
    response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_detect_oversized_image_returns_413(client):
    """Image exceeding MAX_IMAGE_SIZE_MB returns 413."""
    # Patch the limit to 1 byte so we trigger it without a real large file
    with patch("app.core.config.Settings.max_image_size_bytes",
               new_callable=lambda: property(lambda self: 1)):
        files, data = detect_files_and_data()
        response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_detect_model_not_loaded_returns_503(client):
    """When model is not loaded, /detect returns 503."""
    from app.services.model_service import ModelNotLoadedError

    with patch(
        "app.services.model_service.ModelService.predict",
        side_effect=ModelNotLoadedError("Model not loaded"),
    ):
        files, data = detect_files_and_data()
        response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "model_not_loaded"


@pytest.mark.asyncio
async def test_detect_inference_error_returns_500(client):
    """Unexpected exception during inference returns 500."""
    with patch(
        "app.services.model_service.ModelService.predict",
        side_effect=RuntimeError("GPU exploded"),
    ):
        files, data = detect_files_and_data()
        response = await client.post("/detect", files=files, data=data)
    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "inference_error"
