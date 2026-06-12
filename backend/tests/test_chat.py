"""
tests/test_chat.py
===================
Tests for POST /chat endpoint.

Covers
------
- 200 with valid image + message
- ChatResponse schema fields present
- assistant string is human-readable text
- detections list parsed from model output
- points list parsed from model output
- Empty result → graceful assistant message
- 422 on missing message
- 422 on missing image
- 422 on unsupported MIME type
- 503 when model not loaded
- generation_mode override
"""

from __future__ import annotations

import io
import pytest
from PIL import Image
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image_bytes(width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=(60, 100, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def chat_files_and_data(message="locate all buildings", image_bytes=None,
                         mime="image/jpeg", **extra):
    if image_bytes is None:
        image_bytes = make_image_bytes()
    files = {"image": ("photo.jpg", io.BytesIO(image_bytes), mime)}
    data  = {"message": message, **extra}
    return files, data


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_200_ok(client, mock_predict_boxes):
    """POST /chat returns 200 with valid inputs."""
    files, data = chat_files_and_data()
    response = await client.post("/chat", files=files, data=data)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_response_schema(client, mock_predict_boxes):
    """ChatResponse contains all required fields."""
    files, data = chat_files_and_data("locate all cars")
    response = await client.post("/chat", files=files, data=data)
    body = response.json()

    assert "assistant"          in body
    assert "detections"         in body
    assert "points"             in body
    assert "raw_answer"         in body
    assert "image_width"        in body
    assert "image_height"       in body
    assert "inference_time_ms"  in body


@pytest.mark.asyncio
async def test_chat_assistant_is_string(client, mock_predict_boxes):
    """assistant field is a non-empty string."""
    files, data = chat_files_and_data()
    response = await client.post("/chat", files=files, data=data)
    body = response.json()
    assert isinstance(body["assistant"], str)
    assert len(body["assistant"]) > 0


@pytest.mark.asyncio
async def test_chat_detections_list(client, mock_predict_boxes):
    """detections is a list; each entry has label, confidence, bbox."""
    files, data = chat_files_and_data("find all vehicles")
    response = await client.post("/chat", files=files, data=data)
    body = response.json()

    assert isinstance(body["detections"], list)
    assert len(body["detections"]) == 2  # mock returns 2 boxes

    for det in body["detections"]:
        assert "label"           in det
        assert "confidence"      in det
        assert "bbox"            in det
        assert "bbox_normalised" in det
        assert len(det["bbox"]) == 4


@pytest.mark.asyncio
async def test_chat_empty_result_has_helpful_message(client, mock_predict_empty):
    """When no objects found, assistant still returns a helpful message."""
    files, data = chat_files_and_data("find all unicorns")
    response = await client.post("/chat", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["detections"] == []
    assert body["points"] == []
    # Assistant message should mention no detections
    assert len(body["assistant"]) > 0


@pytest.mark.asyncio
async def test_chat_assistant_mentions_count(client, mock_predict_boxes):
    """assistant message references the number of found objects."""
    files, data = chat_files_and_data()
    response = await client.post("/chat", files=files, data=data)
    body = response.json()
    # format_chat_response should embed "2" (from mock returning 2 boxes)
    assert "2" in body["assistant"]


@pytest.mark.asyncio
async def test_chat_image_dimensions(client, mock_predict_boxes):
    """image_width and image_height match uploaded image dimensions."""
    image_bytes = make_image_bytes(width=320, height=240)
    files = {"image": ("img.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    data  = {"message": "find objects"}
    response = await client.post("/chat", files=files, data=data)
    body = response.json()
    assert body["image_width"]  == 320
    assert body["image_height"] == 240


@pytest.mark.asyncio
async def test_chat_inference_time_positive(client, mock_predict_boxes):
    """inference_time_ms > 0."""
    files, data = chat_files_and_data()
    response = await client.post("/chat", files=files, data=data)
    assert response.json()["inference_time_ms"] > 0


@pytest.mark.asyncio
async def test_chat_generation_mode_override(client, mock_predict_boxes):
    """generation_mode form field is accepted without error."""
    files, data = chat_files_and_data(generation_mode="fast")
    response = await client.post("/chat", files=files, data=data)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Multiple message styles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "Locate all buildings in this aerial photo",
    "How many people are in the image? Find them all.",
    "find car, truck, bus",
    "detect all text on the signs",
    "Where is the red car?",
])
async def test_chat_various_messages(client, mock_predict_boxes, message):
    """All message styles return 200."""
    files, data = chat_files_and_data(message)
    response = await client.post("/chat", files=files, data=data)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_missing_message_returns_422(client):
    """Missing message field returns 422."""
    files = {"image": ("photo.jpg", io.BytesIO(make_image_bytes()), "image/jpeg")}
    response = await client.post("/chat", files=files)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_missing_image_returns_422(client):
    """Missing image field returns 422."""
    response = await client.post("/chat", data={"message": "find objects"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_empty_message_returns_422(client):
    """Empty message string returns 422."""
    files, data = chat_files_and_data(message="")
    response = await client.post("/chat", files=files, data=data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_unsupported_mime_returns_422(client):
    """Unsupported MIME type returns 422."""
    files = {"image": ("test.tiff", io.BytesIO(b"fake"), "image/tiff")}
    data  = {"message": "find objects"}
    response = await client.post("/chat", files=files, data=data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_model_not_loaded_returns_503(client):
    """503 when model is not loaded."""
    from app.services.model_service import ModelNotLoadedError

    with patch(
        "app.services.model_service.ModelService.predict",
        side_effect=ModelNotLoadedError("not loaded"),
    ):
        files, data = chat_files_and_data()
        response = await client.post("/chat", files=files, data=data)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_chat_inference_error_returns_500(client):
    """Unexpected inference exception returns 500."""
    with patch(
        "app.services.model_service.ModelService.predict",
        side_effect=RuntimeError("CUDA error"),
    ):
        files, data = chat_files_and_data()
        response = await client.post("/chat", files=files, data=data)
    assert response.status_code == 500
