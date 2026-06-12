"""
app/api/detect.py
==================
POST /detect  —  Object detection and visual grounding endpoint.

Accepts multipart/form-data with:
    image             : uploaded image file
    query             : natural-language detection query
    generation_mode   : optional override ("hybrid" | "fast" | "slow")
    max_new_tokens    : optional token limit override

Returns DetectResponse with structured bounding boxes and points.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.logging import logger
from app.schemas.detection import DetectResponse
from app.services.model_service import ModelNotLoadedError, get_model_service
from app.utils.image_utils import load_and_validate_image
from app.utils.prompt_utils import build_detect_prompt

router = APIRouter()


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect and localise objects in an image",
    description=(
        "Upload an image and a natural-language query. "
        "LocateAnything-3B returns structured bounding boxes and/or points. "
        "Example queries: 'find all cars', 'locate people wearing red shirts', "
        "'find car, person, bicycle'."
    ),
    responses={
        200: {"description": "Detection results with bounding boxes."},
        400: {"description": "Bad request (malformed image or query)."},
        413: {"description": "Image exceeds size limit."},
        422: {"description": "Validation error."},
        503: {"description": "Model not yet loaded."},
    },
)
async def detect(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, BMP)."),
    query: str = Form(
        ...,
        min_length=1,
        max_length=2048,
        description="Natural-language query, e.g. 'find all cars'.",
    ),
    generation_mode: str | None = Form(
        default=None,
        description="Override generation mode: hybrid | fast | slow.",
    ),
    max_new_tokens: int | None = Form(
        default=None,
        ge=64,
        le=8192,
        description="Override max tokens for this request.",
    ),
) -> DetectResponse:
    """
    Detect and localise objects described by a natural-language query.

    The endpoint automatically converts the user query into the correct
    LocateAnything prompt template. Both bounding boxes and points are
    always parsed from the model output.
    """
    logger.info("POST /detect | query={!r} | file={}", query[:80], image.filename)

    # Load and validate image.
    pil_image = await load_and_validate_image(image)
    img_w, img_h = pil_image.size

    # Build the correct LocateAnything prompt from the user query.
    prompt = build_detect_prompt(query)
    logger.debug("Built prompt: {!r}", prompt)

    # Run inference.
    svc = get_model_service()
    try:
        result = svc.predict(
            image=pil_image,
            question=prompt,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
        )
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_not_loaded", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("Inference error on /detect: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "inference_error", "message": f"Model inference failed: {exc}"},
        ) from exc

    raw_answer: str = result["answer"]

    # Always parse both boxes AND points from model output.
    # The model may output either or both depending on the task.
    boxes = svc.parse_boxes(raw_answer, img_w, img_h, label=query)
    points = svc.parse_points(raw_answer, img_w, img_h, label=query)

    logger.info(
        "Detected {} boxes, {} points | time={:.0f}ms",
        len(boxes),
        len(points),
        result["inference_time_ms"],
    )

    return DetectResponse(
        query=query,
        detections=boxes,
        points=points,
        raw_answer=raw_answer,
        image_width=img_w,
        image_height=img_h,
        inference_time_ms=result["inference_time_ms"],
        generation_mode_used=result["generation_mode_used"],
    )
