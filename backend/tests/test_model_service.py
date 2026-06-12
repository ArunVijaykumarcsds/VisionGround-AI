"""
tests/test_model_service.py
============================
Unit tests for ModelService — no GPU, no model download required.

Tests cover:
- Singleton pattern
- parse_boxes(): correct regex, coordinate conversion, degenerate box rejection
- parse_points(): correct regex, coordinate conversion
- _validate_and_resize_image(): RGB conversion, downscale on oversized input
- predict() raises ModelNotLoadedError before load()
"""

from __future__ import annotations

import io
import sys
import types
import functools

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Torch stub (no real GPU/torch needed for these unit tests)
# ---------------------------------------------------------------------------

def _install_torch_stub():
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.bfloat16 = "bfloat16"
        torch_mod.float16  = "float16"
        torch_mod.float32  = "float32"

        class no_grad_ctx:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __call__(self, fn):
                @functools.wraps(fn)
                def w(*a, **k): return fn(*a, **k)
                return w

        torch_mod.no_grad = no_grad_ctx

        class FakeCuda:
            def is_available(self): return False

        torch_mod.cuda = FakeCuda()
        sys.modules["torch"] = torch_mod

    if "psutil" not in sys.modules:
        psutil_mod = types.ModuleType("psutil")
        vm = types.SimpleNamespace(total=32e9, used=8e9, available=24e9)
        psutil_mod.virtual_memory = lambda: vm
        sys.modules["psutil"] = psutil_mod


_install_torch_stub()

from app.services.model_service import (
    ModelService,
    ModelNotLoadedError,
    get_model_service,
)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_same_instance_on_multiple_calls(self):
        s1 = get_model_service()
        s2 = get_model_service()
        assert s1 is s2

    def test_is_loaded_false_before_load(self):
        svc = get_model_service()
        assert svc.is_loaded is False

    def test_predict_raises_before_load(self):
        svc = get_model_service()
        img = Image.new("RGB", (64, 64))
        with pytest.raises(ModelNotLoadedError):
            svc.predict(img, "find objects")


# ---------------------------------------------------------------------------
# parse_boxes
# ---------------------------------------------------------------------------

class TestParseBoxes:

    def test_single_box(self):
        answer = "<box><100><200><400><600></box>"
        boxes = ModelService.parse_boxes(answer, image_width=1000, image_height=1000)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0].bbox
        assert x1 == pytest.approx(100.0)
        assert y1 == pytest.approx(200.0)
        assert x2 == pytest.approx(400.0)
        assert y2 == pytest.approx(600.0)

    def test_two_boxes(self):
        answer = "<box><100><200><400><600></box><box><500><100><800><350></box>"
        boxes = ModelService.parse_boxes(answer, 1000, 1000)
        assert len(boxes) == 2

    def test_zero_boxes_empty_answer(self):
        boxes = ModelService.parse_boxes("no detections", 1000, 1000)
        assert boxes == []

    def test_degenerate_x2_lt_x1_skipped(self):
        answer = "<box><600><100><200><400></box>"
        boxes = ModelService.parse_boxes(answer, 1000, 1000)
        assert boxes == []

    def test_degenerate_y2_lt_y1_skipped(self):
        answer = "<box><100><600><400><200></box>"
        boxes = ModelService.parse_boxes(answer, 1000, 1000)
        assert boxes == []

    def test_full_image_box(self):
        answer = "<box><0><0><1000><1000></box>"
        boxes = ModelService.parse_boxes(answer, 800, 600)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0].bbox
        assert x1 == pytest.approx(0.0)
        assert y1 == pytest.approx(0.0)
        assert x2 == pytest.approx(800.0)
        assert y2 == pytest.approx(600.0)

    def test_coordinate_scaling_non_square(self):
        answer = "<box><500><500><1000><1000></box>"
        boxes = ModelService.parse_boxes(answer, image_width=640, image_height=480)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0].bbox
        assert x1 == pytest.approx(320.0)
        assert y1 == pytest.approx(240.0)
        assert x2 == pytest.approx(640.0)
        assert y2 == pytest.approx(480.0)

    def test_bbox_normalised_in_unit_range(self):
        answer = "<box><250><250><750><750></box>"
        boxes = ModelService.parse_boxes(answer, 1000, 1000)
        assert len(boxes) == 1
        for v in boxes[0].bbox_normalised:
            assert 0.0 <= v <= 1.0

    def test_label_stored(self):
        answer = "<box><0><0><500><500></box>"
        boxes = ModelService.parse_boxes(answer, 1000, 1000, label="car")
        assert boxes[0].label == "car"

    def test_confidence_always_one(self):
        answer = "<box><0><0><500><500></box>"
        boxes = ModelService.parse_boxes(answer, 1000, 1000)
        assert boxes[0].confidence == 1.0

    def test_ten_boxes(self):
        tokens = "".join(
            f"<box><{i*50}><{i*40}><{i*50+100}><{i*40+80}></box>"
            for i in range(10)
        )
        boxes = ModelService.parse_boxes(tokens, 1000, 1000)
        assert len(boxes) == 10

    def test_mixed_valid_and_degenerate(self):
        answer = (
            "<box><100><100><400><400></box>"
            "<box><500><500><200><200></box>"
            "<box><600><100><900><500></box>"
        )
        boxes = ModelService.parse_boxes(answer, 1000, 1000)
        assert len(boxes) == 2


# ---------------------------------------------------------------------------
# parse_points
# ---------------------------------------------------------------------------

class TestParsePoints:

    def test_single_point(self):
        answer = "<box><350><480></box>"
        pts = ModelService.parse_points(answer, 1000, 1000)
        assert len(pts) == 1
        assert pts[0].x == pytest.approx(350.0)
        assert pts[0].y == pytest.approx(480.0)

    def test_two_points(self):
        answer = "<box><100><200></box><box><700><800></box>"
        pts = ModelService.parse_points(answer, 1000, 1000)
        assert len(pts) == 2

    def test_zero_points_empty(self):
        pts = ModelService.parse_points("nothing", 1000, 1000)
        assert pts == []

    def test_coordinate_scaling(self):
        answer = "<box><500><250></box>"
        pts = ModelService.parse_points(answer, 800, 600)
        assert pts[0].x == pytest.approx(400.0)
        assert pts[0].y == pytest.approx(150.0)
        assert pts[0].x_normalised == pytest.approx(0.5)
        assert pts[0].y_normalised == pytest.approx(0.25)

    def test_label_stored(self):
        answer = "<box><500><500></box>"
        pts = ModelService.parse_points(answer, 1000, 1000, label="cat")
        assert pts[0].label == "cat"

    def test_4coord_box_not_parsed_as_point(self):
        """4-coordinate boxes must NOT be matched by the 2-coord point pattern."""
        answer = "<box><100><200><400><600></box>"
        pts = ModelService.parse_points(answer, 1000, 1000)
        assert pts == []


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

class TestImageValidation:

    def setup_method(self):
        self.svc = get_model_service()

    def test_rgba_to_rgb(self):
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
        result = self.svc._validate_and_resize_image(img)
        assert result.mode == "RGB"

    def test_grayscale_to_rgb(self):
        img = Image.new("L", (64, 64))
        result = self.svc._validate_and_resize_image(img)
        assert result.mode == "RGB"

    def test_rgb_unchanged(self):
        img = Image.new("RGB", (64, 64))
        result = self.svc._validate_and_resize_image(img)
        assert result.mode == "RGB"
        assert result.size == (64, 64)

    def test_oversized_downscaled(self):
        img = Image.new("RGB", (4000, 3000))
        result = self.svc._validate_and_resize_image(img)
        assert result.width  <= self.svc.settings.max_image_dimension
        assert result.height <= self.svc.settings.max_image_dimension
        # Aspect ratio preserved
        assert (result.width / result.height) == pytest.approx(4000/3000, rel=0.01)

    def test_small_image_not_upscaled(self):
        img = Image.new("RGB", (100, 80))
        result = self.svc._validate_and_resize_image(img)
        assert result.size == (100, 80)
