"""
app/schemas/detection.py
=========================
Pydantic v2 request and response schemas for all API endpoints.

These schemas serve as the contract between the FastAPI backend and any
client (React frontend, CURL, Python client, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# ===========================================================================
# Detection schemas
# ===========================================================================

class BoundingBox(BaseModel):
    """
    Pixel-coordinate bounding box with normalised confidence.

    Coordinates (x1, y1) represent the top-left corner;
    (x2, y2) represent the bottom-right corner.
    """

    label: str = Field(..., description="Detected object label or phrase.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score in [0, 1].",
    )
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box as [x1, y1, x2, y2] in pixel coordinates.",
    )
    bbox_normalised: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Bounding box as [x1, y1, x2, y2] normalised to [0, 1].",
    )

    @model_validator(mode="after")
    def validate_bbox_ordering(self) -> "BoundingBox":
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(
                f"Invalid bbox: x2 ({x2}) must be ≥ x1 ({x1}) and "
                f"y2 ({y2}) must be ≥ y1 ({y1})."
            )
        return self


class DetectionPoint(BaseModel):
    """
    Pixel-coordinate point output (used for pointing / GUI tasks).
    """

    label: str = Field(..., description="Point label or description.")
    x: float = Field(..., description="X coordinate in pixels.")
    y: float = Field(..., description="Y coordinate in pixels.")
    x_normalised: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y_normalised: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DetectRequest(BaseModel):
    """
    Parameters accompanying a /detect multipart form request.

    The image itself is sent as form data (UploadFile);
    this model covers the text fields.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description=(
            "Natural-language query, e.g. 'find all cars' or "
            "'locate people wearing red shirts'."
        ),
    )
    generation_mode: Optional[str] = Field(
        default=None,
        description="Override generation mode: 'hybrid' | 'fast' | 'slow'.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=8192,
        description="Override max_new_tokens for this request.",
    )


class DetectResponse(BaseModel):
    """
    Structured detection response from /detect.
    """

    query: str = Field(..., description="Echo of the original query.")
    detections: List[BoundingBox] = Field(
        default_factory=list,
        description="List of detected objects with bounding boxes.",
    )
    points: List[DetectionPoint] = Field(
        default_factory=list,
        description="List of point predictions (for pointing tasks).",
    )
    raw_answer: str = Field(
        ...,
        description="Raw model output string before parsing.",
    )
    image_width: int = Field(..., description="Width of the input image in pixels.")
    image_height: int = Field(..., description="Height of the input image in pixels.")
    inference_time_ms: float = Field(
        ...,
        description="Server-side inference time in milliseconds.",
    )
    generation_mode_used: str = Field(
        ...,
        description="Generation mode that was actually used.",
    )


# ===========================================================================
# Chat schemas
# ===========================================================================

class ChatMessage(BaseModel):
    """Single message in a chat conversation."""

    role: str = Field(
        ...,
        pattern="^(user|assistant|system)$",
        description="Message role: 'user', 'assistant', or 'system'.",
    )
    content: str = Field(..., min_length=1, description="Message text content.")


class ChatRequest(BaseModel):
    """
    Parameters for /chat endpoint.

    Image is sent as multipart form data alongside these text fields.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User message, e.g. 'Locate all buildings in this image'.",
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional prior conversation history for multi-turn chat.",
    )
    generation_mode: Optional[str] = Field(
        default=None,
        description="Override generation mode: 'hybrid' | 'fast' | 'slow'.",
    )


class ChatResponse(BaseModel):
    """Chat-style response from /chat."""

    assistant: str = Field(
        ...,
        description="Assistant reply with embedded detection summary.",
    )
    detections: List[BoundingBox] = Field(
        default_factory=list,
        description="Parsed detections (if any) from the model response.",
    )
    points: List[DetectionPoint] = Field(
        default_factory=list,
        description="Parsed point predictions (if any).",
    )
    raw_answer: str = Field(
        ...,
        description="Raw model output before post-processing.",
    )
    image_width: int
    image_height: int
    inference_time_ms: float


# ===========================================================================
# Health schema
# ===========================================================================

class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = Field(default="ok")
    model_loaded: bool = Field(..., description="Whether the model is ready.")
    device: str = Field(..., description="Device the model is running on.")
    model_path: str = Field(..., description="Model identifier or local path.")
    memory_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional GPU/CPU memory diagnostics.",
    )


# ===========================================================================
# Error schema
# ===========================================================================

class ErrorDetail(BaseModel):
    """Standardised error payload."""

    error: str = Field(..., description="Short error code or type.")
    message: str = Field(..., description="Human-readable error description.")
    detail: Optional[Any] = Field(default=None, description="Extra debug context.")
