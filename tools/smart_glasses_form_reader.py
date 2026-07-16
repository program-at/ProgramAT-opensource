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
  Reading mode (fillable field)  : "Reading mode. Field: student name."
  Reading mode (sub-field)       : "Reading mode. Chair/Co-Chairs, Title."
  Reading mode (sub-field long)  : "Reading mode. Field: Chair/Co-Chairs, Name and UM email address."
  Reading mode (heading)         : "Reading mode. Dissertation Committee Request Form."
  Writing mode (move left)       : "Writing mode. Move left, student name."
  Writing mode (far right)       : "Writing mode. Move left, far from start of student name."
  Writing mode (ready)           : "Writing mode. Ready to write, student name."
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
    "A form row may contain multiple separate fields side by side. Each field "
    "has its own fillable line or box. The start of a field is the left edge "
    "of THAT field's own line or box — not the left edge of the page.\n"
    "IMPORTANT spatial convention: on paper forms, a person writes BESIDE or "
    "ABOVE the field's line — never below it. The pen tip should be just above "
    "or at the same vertical level as the line, not underneath it. Use this "
    "to identify which field the pen is targeting: the field whose line is "
    "immediately below or level with the pen tip.\n"
    "The form is in English, so writing begins at the left edge of each "
    "individual field. Guide the pen relative to the specific field it is "
    "nearest to. Use this scale:\n"
    "- Pen is near the left portion of that field's line or box (and at or "
    "above the line): say 'Writing mode. Ready to write, <field name>.'\n"
    "- Pen is in the middle of that field: say 'Writing mode. Move left, <field name>.'\n"
    "- Pen is far to the right within that field: say 'Writing mode. Move left, far from "
    "start of <field name>.'\n"
    "- Pen is not yet over any field: begin with 'Writing mode.' then give a direction cue that matches the "
    "actual position of the nearest field relative to the pen — say 'Move down' "
    "only if the field is physically below the pen, 'Move up' if above, "
    "'Move right' if to the right, 'Move left' if to the left, or combine "
    "directions if needed (e.g. 'Writing mode. Move right and down to <field name>.').\n"
    "Always name the specific field the pen is targeting.\n\n"
    "READING MODE instructions:\n"
    "Begin every reading mode response with 'Reading mode.' followed by the content.\n"
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
    "Reply in 15 words or fewer. Do not include any other labels or explanations beyond the mode prefix."
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
