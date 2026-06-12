"""
tests/test_schemas.py
======================
Unit tests for all Pydantic schemas in app/schemas/detection.py.
"""

from __future__ import annotations
import pytest
from pydantic import ValidationError
from app.schemas.detection import (
    BoundingBox, ChatMessage, ChatResponse, DetectResponse,
    DetectionPoint, ErrorDetail, HealthResponse,
)


class TestBoundingBox:
    def test_valid(self):
        b = BoundingBox(label="car", confidence=0.95, bbox=[10.0, 20.0, 100.0, 200.0])
        assert b.label == "car"
        assert b.bbox == [10.0, 20.0, 100.0, 200.0]

    def test_with_normalised(self):
        b = BoundingBox(label="car", confidence=1.0,
                        bbox=[100.0, 200.0, 400.0, 600.0],
                        bbox_normalised=[0.1, 0.2, 0.4, 0.6])
        assert b.bbox_normalised == [0.1, 0.2, 0.4, 0.6]

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(label="x", confidence=1.5, bbox=[0, 0, 1, 1])

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(label="x", confidence=-0.1, bbox=[0, 0, 1, 1])

    def test_degenerate_x2_lt_x1_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(label="x", confidence=1.0, bbox=[100, 10, 50, 80])

    def test_degenerate_y2_lt_y1_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(label="x", confidence=1.0, bbox=[10, 100, 80, 50])

    def test_bbox_requires_4_elements(self):
        with pytest.raises(ValidationError):
            BoundingBox(label="x", confidence=1.0, bbox=[10, 20, 30])

    def test_normalised_optional(self):
        b = BoundingBox(label="x", confidence=1.0, bbox=[0, 0, 100, 100])
        assert b.bbox_normalised is None

    def test_serialisation(self):
        b = BoundingBox(label="car", confidence=0.9, bbox=[1.0, 2.0, 3.0, 4.0])
        d = b.model_dump()
        assert d["label"] == "car"
        assert d["bbox"] == [1.0, 2.0, 3.0, 4.0]


class TestDetectionPoint:
    def test_valid(self):
        pt = DetectionPoint(label="cat", x=350.0, y=480.0)
        assert pt.x == pytest.approx(350.0)

    def test_normalised_above_one_rejected(self):
        with pytest.raises(ValidationError):
            DetectionPoint(label="x", x=0, y=0, x_normalised=1.5)

    def test_normalised_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            DetectionPoint(label="x", x=0, y=0, y_normalised=-0.1)

    def test_normalised_optional(self):
        pt = DetectionPoint(label="x", x=100.0, y=200.0)
        assert pt.x_normalised is None


class TestDetectResponse:
    def _make(self, **overrides):
        defaults = dict(
            query="find all cars", detections=[], points=[],
            raw_answer="raw", image_width=1000, image_height=1000,
            inference_time_ms=234.5, generation_mode_used="hybrid",
        )
        defaults.update(overrides)
        return DetectResponse(**defaults)

    def test_basic(self):
        r = self._make()
        assert r.query == "find all cars"
        assert r.inference_time_ms == pytest.approx(234.5)

    def test_with_detections(self):
        box = BoundingBox(label="car", confidence=1.0, bbox=[10, 20, 100, 200])
        r = self._make(detections=[box])
        assert len(r.detections) == 1


class TestChatResponse:
    def test_basic(self):
        r = ChatResponse(
            assistant="I found 2 cars.", detections=[], points=[],
            raw_answer="raw", image_width=800, image_height=600,
            inference_time_ms=187.3,
        )
        assert r.assistant == "I found 2 cars."

    def test_assistant_required(self):
        with pytest.raises(ValidationError):
            ChatResponse(
                detections=[], points=[], raw_answer="",
                image_width=100, image_height=100, inference_time_ms=0,
            )


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse(model_loaded=True, device="cuda",
                            model_path="nvidia/LocateAnything-3B")
        assert h.status == "ok"

    def test_memory_info_optional(self):
        h = HealthResponse(model_loaded=False, device="cpu",
                            model_path="nvidia/LocateAnything-3B")
        assert h.memory_info is None


class TestChatMessage:
    def test_valid_roles(self):
        for role in ("user", "assistant", "system"):
            m = ChatMessage(role=role, content="hello")
            assert m.role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="bot", content="hello")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")


class TestErrorDetail:
    def test_construction(self):
        e = ErrorDetail(error="model_not_loaded", message="Not ready.")
        assert e.error == "model_not_loaded"
        assert e.detail is None

    def test_with_detail(self):
        e = ErrorDetail(error="err", message="msg", detail={"key": "val"})
        assert e.detail["key"] == "val"
