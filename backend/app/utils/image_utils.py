"""
app/utils/image_utils.py
=========================
Utility functions for image validation, loading, and conversion.

These helpers are used by the API route handlers before passing images
to the ModelService. They provide clear error messages to clients when
uploads do not meet requirements.
"""

from __future__ import annotations

import io
from typing import Tuple

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.core.logging import logger
from app.services.model_service import ImageValidationError


async def load_and_validate_image(upload: UploadFile) -> Image.Image:
    """
    Read an UploadFile, validate it, and return a PIL Image.

    Validation checks:
        1. MIME type is in the allowed list.
        2. File size does not exceed MAX_IMAGE_SIZE_MB.
        3. File can be decoded as a valid image.

    Raises HTTPException(422) on validation failure.
    """
    settings = get_settings()

    # --- MIME type check ---
    content_type = upload.content_type or ""
    if content_type not in settings.allowed_image_types_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "unsupported_image_type",
                "message": (
                    f"Content-Type '{content_type}' is not supported. "
                    f"Allowed types: {settings.allowed_image_types_list}"
                ),
            },
        )

    # --- Read bytes ---
    try:
        raw_bytes = await upload.read()
    except Exception as exc:
        logger.exception("Failed to read upload: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "read_error", "message": "Could not read uploaded file."},
        ) from exc

    # --- Size check ---
    size_bytes = len(raw_bytes)
    if size_bytes > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "image_too_large",
                "message": (
                    f"Image size {size_bytes / 1e6:.1f} MB exceeds the "
                    f"maximum allowed {settings.max_image_size_mb} MB."
                ),
            },
        )

    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "empty_file", "message": "Uploaded file is empty."},
        )

    # --- Decode image ---
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # force decode to catch truncated files
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "invalid_image",
                "message": "File could not be decoded as a valid image.",
            },
        ) from exc
    except Exception as exc:
        logger.exception("Image decode error: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "image_decode_error",
                "message": f"Image decoding failed: {exc}",
            },
        ) from exc

    logger.debug(
        "Image validated | size={:.1f}KB | mode={} | dims={}x{}",
        size_bytes / 1024,
        image.mode,
        image.width,
        image.height,
    )

    return image


def image_to_bytes(image: Image.Image, fmt: str = "JPEG", quality: int = 85) -> bytes:
    """Encode a PIL Image to bytes in the given format."""
    buf = io.BytesIO()
    save_params: dict = {}
    if fmt.upper() == "JPEG":
        save_params["quality"] = quality
        if image.mode != "RGB":
            image = image.convert("RGB")
    image.save(buf, format=fmt, **save_params)
    return buf.getvalue()


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """Clamp bounding-box coordinates to image boundaries."""
    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_height))
    return x1, y1, x2, y2
