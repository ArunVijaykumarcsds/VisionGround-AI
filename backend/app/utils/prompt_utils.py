"""
app/utils/prompt_utils.py
==========================
Helpers for converting natural-language user queries into the exact
prompt templates expected by LocateAnything-3B.

Prompt templates are taken verbatim from the official model card to
ensure maximum compatibility with the model's training distribution.
"""

from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------------------------
# Task-to-prompt mapping
# ---------------------------------------------------------------------------

def build_detect_prompt(query: str) -> str:
    """
    Convert a free-form detection query to the model's detection prompt.

    Handles two cases:
    - Category list: "find cars, people, bicycles"  →
      "Locate all the instances that matches the following description:
       cars</c>people</c>bicycles."
    - Free-form phrase: "red cars near a fire hydrant"  →
      "Locate all the instances that match the following description:
       red cars near a fire hydrant."
    """
    # Heuristic: if the query looks like comma-separated short nouns, treat
    # as a category list and join with </c>.
    stripped = query.strip().rstrip(".")
    parts = [p.strip() for p in stripped.split(",") if p.strip()]

    if len(parts) > 1 and all(len(p.split()) <= 4 for p in parts):
        # Category list mode.
        cats = "</c>".join(parts)
        return (
            f"Locate all the instances that matches the following description: {cats}."
        )

    # Free-form multi-instance phrase.
    return (
        f"Locate all the instances that match the following description: {stripped}."
    )


def build_single_ground_prompt(phrase: str) -> str:
    """Phrase grounding prompt for a single instance."""
    phrase = phrase.strip().rstrip(".")
    return (
        f"Locate a single instance that matches the following description: {phrase}."
    )


def build_text_detection_prompt() -> str:
    """Scene text detection prompt."""
    return "Detect all the text in box format."


def build_gui_ground_prompt(phrase: str, output_type: str = "box") -> str:
    """GUI grounding prompt (box or point mode)."""
    phrase = phrase.strip().rstrip(".")
    if output_type == "point":
        return f"Point to: {phrase}."
    return f"Locate the region that matches the following description: {phrase}."


def build_point_prompt(phrase: str) -> str:
    """Pointing task prompt."""
    phrase = phrase.strip().rstrip(".")
    return f"Point to: {phrase}."


def build_text_ground_prompt(phrase: str) -> str:
    """Text grounding prompt."""
    phrase = phrase.strip().rstrip(".")
    return f"Please locate the text referred as {phrase}."


def classify_query(query: str) -> str:
    """
    Heuristically classify a free-form user query into a task type.

    Returns one of:
        "detect"          – multi-object detection
        "text_detection"  – scene text / OCR detection
        "gui"             – GUI element grounding
        "point"           – pointing task
        "ground"          – phrase grounding (default)
    """
    lower = query.lower()

    if any(kw in lower for kw in ("detect all text", "find text", "ocr", "read text")):
        return "text_detection"

    if any(kw in lower for kw in ("click", "button", "icon", "gui", "interface")):
        return "gui"

    if lower.startswith("point to") or lower.startswith("point at"):
        return "point"

    # Default to general detection for most queries.
    return "detect"


def format_chat_response(raw_answer: str, n_boxes: int, n_points: int) -> str:
    """
    Generate a human-readable assistant message from model output stats.

    Used by the /chat endpoint to produce a conversational summary.
    """
    if n_boxes == 0 and n_points == 0:
        return (
            "I processed your image and query, but did not detect any matching "
            "objects. Try rephrasing your query or use a more specific description."
        )

    parts: List[str] = []
    if n_boxes:
        noun = "object" if n_boxes == 1 else "objects"
        parts.append(f"I found {n_boxes} {noun}")
    if n_points:
        noun = "location" if n_points == 1 else "locations"
        parts.append(f"identified {n_points} {noun}")

    summary = " and ".join(parts) + " in the image."
    return summary[0].upper() + summary[1:]
