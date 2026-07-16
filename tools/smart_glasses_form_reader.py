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
  Reading mode (top-level field) : "Reading field student name."
  Reading mode (sub-field)       : "Chair/Co-Chairs, Title."
  Reading mode (sub-field long)  : "Field: Chair/Co-Chairs, Name and UM email address."
  Writing mode                   : "Slightly left to field student name."
  Centered                       : "Writing centered on field student name, ready to write."
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
    "Determine where the pen tip is positioned relative to the nearest form "
    "input field. If the pen is centered over a field say: "
    "'Writing centered on field <field name>, ready to write.' "
    "Otherwise say the direction the pen must move and the field name, e.g. "
    "'Slightly left to field student name.'\n\n"
    "READING MODE instructions:\n"
    "Read the text or field label directly under or nearest to the pointing "
    "finger. If the field belongs to a named section or group on the form, "
    "always include the section name before the field name. "
    "Examples: 'Field: Chair/Co-Chairs, Name and UM email address.' or "
    "'Chair/Co-Chairs, Title.' "
    "If the finger points to a plain text area (not a labelled field) say: 'Reading <text>.' "
    "For multiple fields in the same row read them left to right, each with "
    "its section name if applicable. "
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
