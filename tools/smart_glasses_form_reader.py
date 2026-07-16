"""
Smart Glasses Form Reader

Assists blind or low vision users wearing smart glasses to read and fill out
paper forms using a finger or pen.

Operates in two modes based on what is visible in the camera frame:
- Reading mode (finger only): identifies text or field labels under the finger
- Writing mode (pen present): provides spatial directions to align pen with the
  correct input field

In live/streaming mode responses are capped at 15 words.

Example outputs:
  Reading mode (fillable field)  : "Field: student name."
  Reading mode (sub-field)       : "Chair/Co-Chairs, Title."
  Reading mode (sub-field long)  : "Field: Chair/Co-Chairs, Name and UM email address."
  Reading mode (heading)         : "Dissertation Committee Request Form."
  Writing mode (move right)      : "Move right to start of field student name."
  Writing mode (at start)        : "At start of field student name, ready to write."
"""

import numpy as np
from typing import Any, Dict, Optional

from model_router_client import copilot_llm_call

TOOL_NAME = "smart_glasses_form_reader"
STREAMING_WORD_LIMIT = 15

TOOL_PROMPT = (
    "You are an assistant for a blind user wearing smart glasses who is "
    "interacting with a paper form.\n\n"
    "Look at the image and decide the current mode:\n"
    "- WRITING MODE: a pen, pencil, or other writing instrument is visible "
    "alongside a hand or finger.\n"
    "- READING MODE: only a finger or hand is visible (no pen).\n\n"
    "WRITING MODE instructions:\n"
    "The goal is to position the pen at the LEFT-MOST edge of the nearest "
    "fillable line or box (the form is in English, so writing starts at the "
    "left). Determine where the pen tip is relative to that left edge:\n"
    "- If the pen is already at (or very close to) the left start of the field "
    "say: 'At start of field <field name>, ready to write.'\n"
    "- Otherwise, give a short direction and distance cue toward the left edge, "
    "e.g. 'Move right to start of field student name.' or "
    "'Slightly left, near start of field student name.'\n"
    "Always name the field the pen is targeting.\n\n"
    "READING MODE instructions:\n"
    "Read the text or label directly under or nearest to the pointing finger.\n"
    "- Use the word 'field' ONLY for actual fillable elements: blank input "
    "lines, text boxes, or checkboxes where the user is meant to write "
    "something. Example: 'Field: Chair/Co-Chairs, Name and UM email address.' "
    "or 'Chair/Co-Chairs, Title.'\n"
    "- Do NOT use the word 'field' for headings, section titles, instructions, "
    "or any other text that is not a place the user fills in. For those, just "
    "read the text aloud. Example: 'Dissertation Committee Request Form.' or "
    "'Computer Science and Engineering Graduate Program.'\n"
    "If the pointed element belongs to a named section or group, always include "
    "the section name before the element name.\n"
    "For multiple fillable fields in the same row read them left to right.\n"
    "If the form cannot be read (e.g. due to poor lighting) say: "
    "'Cannot read form: <reason>.'\n\n"
    "Reply in 15 words or fewer. Do not include mode labels or explanations."
)


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
        result = copilot_llm_call(
            capability="general_reasoning",
            goal=(
                "Read the paper form or guide pen alignment for a blind user "
                "based on whether a pen or only a finger is visible."
            ),
            messages=[{"role": "user", "content": TOOL_PROMPT}],
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "smart glasses form reader: detect pen or finger and either "
                    "read form text or provide pen alignment guidance"
                ),
            },
        )

        response = (result.get("response") or "").strip()
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
