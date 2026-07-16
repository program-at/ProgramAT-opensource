"""
Smart Glasses Form Reader

Assists blind or low vision users wearing smart glasses to read and fill out
paper forms using a finger or pen.

Operates in two modes based on what is visible in the camera frame:
- Reading mode  (finger only): identifies text or field labels under the finger
- Writing mode  (pen present): provides spatial directions to align pen with the
  correct input field

In live/streaming mode responses are capped at 15 words.

Example outputs:
  Reading mode : "Reading field student name."
  Writing mode : "Slightly left to field student name."
  Centered     : "Writing centered on field student name, ready to write."
"""

import numpy as np
from typing import Any, Dict, Optional

from model_router_client import copilot_llm_call

TOOL_NAME = "smart_glasses_form_reader"
STREAMING_WORD_LIMIT = 15

_PEN_LABELS = [
    "pen", "pencil", "marker", "ballpoint pen", "writing instrument", "stylus",
]
_HAND_LABELS = [
    "finger", "hand", "index finger", "pointing finger",
]
_ALL_LABELS = _PEN_LABELS + _HAND_LABELS


def _trim(text: str, limit: int = STREAMING_WORD_LIMIT) -> str:
    """Return text trimmed to at most *limit* words."""
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit])


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Entry point for the Smart Glasses Form Reader tool.

    Args:
        image:      Camera frame as a NumPy array (BGR, from OpenCV). May be None.
        input_data: Optional configuration dict (currently unused; reserved for
                    future per-session overrides).

    Returns:
        A plain str spoken via TTS, or a dict with ``audio`` / ``text`` keys.
        Returns ``""`` when there is nothing new to say (streaming suppression).
    """
    if image is None:
        return {
            "audio": {
                "type": "error",
                "text": "No camera image available for form reader.",
                "interrupt": False,
            },
            "text": "No camera image available for form reader.",
        }

    if input_data is None:
        input_data = {}

    try:
        # ------------------------------------------------------------------
        # Stage 1 – detect finger and pen to determine the operating mode
        # ------------------------------------------------------------------
        detection_result = copilot_llm_call(
            capability="object_detection_localization",
            goal=(
                "Detect any fingers, hands, pens, pencils, markers, or writing "
                "instruments visible in this image of a paper form."
            ),
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "detect finger hand and pen or writing instrument on paper form",
                "target_labels": _ALL_LABELS,
            },
        )

        detection_artifact = detection_result.get("artifact") or {}
        detections = (
            detection_artifact.get("detections", [])
            if isinstance(detection_artifact, dict)
            else []
        )

        # Pen present → writing mode; finger only → reading mode
        pen_present = any(
            str(d.get("label", "")).lower() in {lbl.lower() for lbl in _PEN_LABELS}
            for d in detections
        )

        # ------------------------------------------------------------------
        # Stage 2a – Writing mode: spatial guidance for pen alignment
        # ------------------------------------------------------------------
        if pen_present:
            stage2 = copilot_llm_call(
                capability="spatial_reasoning",
                goal=(
                    "A user is filling out a paper form with a pen. "
                    "Determine where the pen tip is positioned relative to the "
                    "form's input fields. Say whether the pen is centered on a "
                    "field, or needs to move left, right, up, or down. "
                    "Name the specific field (e.g. 'student name', 'ID'). "
                    "If the pen is centered, say 'Writing centered on field "
                    "<field name>, ready to write.' "
                    "Otherwise say the direction and the field name, e.g. "
                    "'Slightly left to field student name.' "
                    "Reply in 15 words or fewer."
                ),
                images=[image],
                metadata={
                    "tool_name": TOOL_NAME,
                    "route_text": "spatial pen alignment guidance for paper form field",
                    "previous_stage_artifact": detection_artifact,
                },
            )
            response = (stage2.get("response") or "").strip()
            if not response:
                return ""
            trimmed = _trim(response)
            return {
                "audio": {
                    "type": "speech",
                    "text": trimmed,
                    "rate": 1.0,
                    "interrupt": True,
                },
                "text": trimmed,
            }

        # ------------------------------------------------------------------
        # Stage 2b – Reading mode: OCR the text/label under the finger
        # ------------------------------------------------------------------
        stage2 = copilot_llm_call(
            capability="ocr",
            goal=(
                "A blind user is pointing at a paper form with their finger. "
                "Read the text or field label that is directly under or nearest "
                "to the pointing finger. "
                "If the finger points to a plain text area, say 'Reading <text>.' "
                "If the finger points to a form field label, say 'Reading field "
                "<label>.' or 'Field <label>.' "
                "For multiple fields in the same row read them left to right. "
                "If the form cannot be read (e.g. due to poor lighting) say "
                "'Cannot read form: <reason>.' "
                "Reply in 15 words or fewer."
            ),
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "read text or field label under pointing finger on paper form",
                "previous_stage_artifact": detection_artifact,
            },
        )
        response = (stage2.get("response") or "").strip()
        if not response:
            return ""
        trimmed = _trim(response)
        return {
            "audio": {
                "type": "speech",
                "text": trimmed,
                "rate": 1.0,
                "interrupt": True,
            },
            "text": trimmed,
        }

    except Exception as exc:
        error_msg = f"Form reader error: {str(exc)[:60]}"
        return {
            "audio": {"type": "error", "text": error_msg, "interrupt": False},
            "text": error_msg,
        }
