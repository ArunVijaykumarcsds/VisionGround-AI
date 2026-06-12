"""
app/api/chat.py
================
POST /chat  —  Conversational grounding endpoint.

Accepts an image + free-form message and returns a chat-style response
that includes a human-readable summary and structured detections.

The endpoint supports optional conversation history for multi-turn chat.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.logging import logger
from app.schemas.detection import ChatResponse
from app.services.model_service import ModelNotLoadedError, get_model_service
from app.utils.image_utils import load_and_validate_image
from app.utils.prompt_utils import build_detect_prompt, format_chat_response

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat-style visual grounding",
    description=(
        "Upload an image and a natural-language message. "
        "Returns a conversational assistant reply plus structured detections. "
        "Example: 'Locate all buildings in this aerial image.'"
    ),
    responses={
        200: {"description": "Chat response with detection summary."},
        400: {"description": "Bad request."},
        413: {"description": "Image exceeds size limit."},
        422: {"description": "Validation error."},
        503: {"description": "Model not yet loaded."},
    },
)
async def chat(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, BMP)."),
    message: str = Form(
        ...,
        min_length=1,
        max_length=4096,
        description="User message, e.g. 'Locate all buildings in this image'.",
    ),
    generation_mode: str | None = Form(
        default=None,
        description="Override generation mode: hybrid | fast | slow.",
    ),
) -> ChatResponse:
    """
    Chat-style visual grounding endpoint.

    Converts the user message into an appropriate LocateAnything prompt,
    runs inference, parses structured outputs, and wraps them in a
    natural-language assistant reply suitable for a chat UI.
    """
    logger.info("POST /chat | message={!r} | file={}", message[:80], image.filename)

    # Load and validate image.
    pil_image = await load_and_validate_image(image)
    img_w, img_h = pil_image.size

    # Build LocateAnything prompt from the chat message.
    prompt = build_detect_prompt(message)
    logger.debug("Built prompt: {!r}", prompt)

    # Run inference.
    svc = get_model_service()
    try:
        result = svc.predict(
            image=pil_image,
            question=prompt,
            generation_mode=generation_mode,
        )
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_not_loaded", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("Inference error on /chat: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "inference_error", "message": f"Model inference failed: {exc}"},
        ) from exc

    raw_answer: str = result["answer"]

    # Parse boxes and points from model output.
    boxes = svc.parse_boxes(raw_answer, img_w, img_h, label=message)
    points = svc.parse_points(raw_answer, img_w, img_h, label=message)

    # Build conversational assistant reply.
    assistant_msg = format_chat_response(raw_answer, len(boxes), len(points))

    logger.info(
        "Chat response | {} boxes, {} points | time={:.0f}ms",
        len(boxes),
        len(points),
        result["inference_time_ms"],
    )

    return ChatResponse(
        assistant=assistant_msg,
        detections=boxes,
        points=points,
        raw_answer=raw_answer,
        image_width=img_w,
        image_height=img_h,
        inference_time_ms=result["inference_time_ms"],
    )
