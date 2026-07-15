"""
Physical Interface Accessibility Aid

Helps blind users navigate physical interfaces (keypads, buttons, touchscreens,
thermostats, kiosks, checkout terminals, etc.) by providing real-time directional
guidance to locate and identify buttons.

Single-call approach: one copilot_llm_call handles interface identification,
spatial reasoning, and guidance generation together, with all output rules
enforced in that prompt.

Live mode: responds at most every 1.5 seconds; returns "" when nothing has
changed so the same phrase is not repeated continuously. Streaming responses
are capped at 10 words via both system-prompt instruction and code enforcement.
Output format (cardinal-only directions, no banned verbs) is also enforced
in code so restrictions apply regardless of whether the backend's planning
and routing pipeline is enabled.
"""

import re
import time
import numpy as np
from typing import Any, Dict, Optional

from model_router_client import copilot_llm_call

TOOL_NAME = "physical_interface_aid"

# Maximum words in a streaming/live-mode response (enforced in code + via prompt)
STREAMING_WORD_LIMIT = 10

# Maximum characters for the raw error message portion (truncated at word boundary)
MAX_ERROR_MSG_LEN = 150

# Minimum seconds between spoken responses in live mode.
MIN_RESPONSE_INTERVAL = 1.5

# Re-announce the same guidance after this many frames even when the scene is
# unchanged, so the user hears confirmation and doesn't experience prolonged
# silence while the server is actively processing frames.
REPEAT_INTERVAL = 10

# ── Code-level output-format enforcement (applied regardless of routing mode) ─
# Diagonal directions → nearest cardinal direction.  Named groups:
#   vert  — the vertical component (up/upper → "up"; down/lower → "down")
#   horiz — the horizontal component (left/right, discarded)
# The separator between vert and horiz is optional and may be a hyphen or space.
_DIAGONAL_RE = re.compile(
    r"\b(?P<vert>up(?:per)?|down|lower)(?:-|\s)?(?P<horiz>left|right)\b",
    re.IGNORECASE,
)

# Banned verbs — whole-word match including common inflections.
_BANNED_VERB_RE = re.compile(
    r"\b(touch(?:ing)?|tap(?:ping)?|press(?:ing)?|reach(?:ing)?"
    r"|find(?:ing)?|locat(?:e|ing))\b",
    re.IGNORECASE,
)


def _fix_diagonals(text: str) -> str:
    """Replace diagonal directions with their primary cardinal equivalent."""
    def _replace(m: re.Match) -> str:
        vert = m.group("vert").lower()
        return "down" if vert in ("down", "lower") else "up"

    return _DIAGONAL_RE.sub(_replace, text)


def _enforce_output_format(text: str) -> str:
    """
    Code-level safety net that enforces the allowed output format, regardless
    of whether the backend's planning/routing pipeline is enabled.

    Applied after the LLM response is received:
      1. Diagonal directions are replaced with cardinal equivalents.
      2. If the result still contains a banned verb, suppress it (return "")
         so non-compliant guidance is never spoken to the user.
    """
    cleaned = _fix_diagonals(text)
    if _BANNED_VERB_RE.search(cleaned):
        return ""
    return cleaned

# ── streaming state ──────────────────────────────────────────────────────────
_last_response: str = ""
_frame_count: int = 0
_last_spoken_time: float = 0.0


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Main entry point for the physical interface accessibility aid.

    Args:
        image:      Camera frame as a numpy array (BGR, from OpenCV). May be None.
        input_data: Optional config dict (currently unused; reserved for future
                    options such as a target button name).

    Returns:
        A spoken string (≤ 10 words in streaming/live mode) directing the user
        to or confirming the button their finger is on.  Returns "" when the
        scene has not changed or when called too soon (< 1.5 s since last
        spoken output), so TTS is not triggered redundantly.
        Returns an audio-error dict when no image is available.
    """
    global _last_response, _frame_count, _last_spoken_time
    _frame_count += 1

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {
                "type": "error",
                "text": "No camera image available for interface navigation.",
                "rate": 1.0,
                "interrupt": False,
            },
            "text": "No camera image available.",
        }

    if input_data is None:
        input_data = {}

    try:
        # ── Single call: identify interface, reason spatially, produce guidance ──
        guidance_result = copilot_llm_call(
            capability="general_reasoning",
            goal=(
                "Look at the physical interface and the user's finger in the image. "
                "Output EXACTLY ONE of these three forms (≤10 words total, "
                "cardinal directions only — left, right, up, or down): "
                "A) 'your finger is on [element]' — only if the finger clearly "
                "overlaps the button area. "
                "B) 'move [direction] towards [element]' — if the finger is near "
                "but not on a button. Direction: left, right, up, or down only. "
                "C) '[element a] is slightly [direction] of your finger, "
                "[element b] is slightly [opposite direction] of your finger' — "
                "only when equidistant to two buttons. "
                "Only name buttons/controls the user can physically activate. "
                "Never mention displays, readouts, clocks, or non-interactive elements. "
                "Never use: touch, tap, press, reach, find, locate."
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are providing real-time audio guidance for a blind user "
                        "navigating a physical interface (keypad, thermostat, kiosk, "
                        "checkout terminal, etc.). The user cannot see — audio is "
                        "their only feedback.\n\n"
                        "From the image, simultaneously: identify all interactive "
                        "buttons or controls (NOT displays, screens, digital readouts, "
                        "clocks, timers, or status indicators), locate the user's "
                        "finger, determine whether it directly overlaps a button or "
                        "is only nearby, and produce the spoken guidance.\n\n"
                        "You MUST output EXACTLY ONE of these three forms "
                        "with ≤10 words total:\n"
                        "  A) 'your finger is on [element]' — only when the finger "
                        "clearly overlaps the button area.\n"
                        "  B) 'move [direction] towards [element]' — when the finger "
                        "is near but NOT directly on a button. Direction must be "
                        "exactly one word: left, right, up, or down. No diagonals.\n"
                        "  C) '[element a] is slightly [direction] of your finger, "
                        "[element b] is slightly [opposite direction] of your finger' "
                        "— only when the finger is genuinely equidistant between two "
                        "elements.\n\n"
                        "RULES (enforced in all cases):\n"
                        "- Reference only the single most relevant button or control.\n"
                        "- Never mention displays, readouts, clocks, timers, status "
                        "indicators, or any non-interactive element.\n"
                        "- Never use: touch, touching, tap, tapping, press, pressing, "
                        "reach, reaching, find, finding, locate, locating.\n"
                        "- Never use diagonal directions.\n"
                        "- If no finger is visible, say 'no finger visible'.\n"
                        "- If lighting is too poor to determine position, say "
                        "'lighting too poor to guide'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "What physical interface is this? Where is the user's finger "
                        "relative to the buttons? Give me the spoken guidance now."
                    ),
                },
            ],
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "guide blind user's finger on physical interface with directional audio"
                ),
            },
        )

        response: str = guidance_result.get("response", "").strip()

        if not response:
            return ""

        # ── Enforce 10-word cap (safety net on top of prompt instruction) ────
        words = response.split()
        if len(words) > STREAMING_WORD_LIMIT:
            response = " ".join(words[:STREAMING_WORD_LIMIT])

        # ── Enforce output format (safety net regardless of routing mode) ─────
        # Replaces diagonal directions with cardinal equivalents and suppresses
        # responses containing banned verbs.  Applied here in code so the
        # restriction holds whether or not the backend's planning/routing
        # pipeline is active.
        response = _enforce_output_format(response)
        if not response:
            return ""

        # ── Streaming deduplication ───────────────────────────────────────────
        # Suppress repeat announcements when the scene hasn't changed, but
        # re-announce every REPEAT_INTERVAL frames so the user isn't left in
        # silence while the server is actively processing (the cause of the
        # "no output" symptom reported when server logs show a valid response).
        if response == _last_response and _frame_count % REPEAT_INTERVAL != 0:
            return ""

        # ── Rate limit: at most one spoken response every 1.5 seconds ────────
        now = time.monotonic()
        if now - _last_spoken_time < MIN_RESPONSE_INTERVAL:
            return ""

        _last_response = response
        _last_spoken_time = now
        return response

    except Exception as exc:  # catch all exceptions and return audio error feedback
        raw_msg = str(exc)
        if len(raw_msg) > MAX_ERROR_MSG_LEN:
            # Truncate at a word boundary.  raw_msg[:MAX_ERROR_MSG_LEN] gives
            # ≤MAX_ERROR_MSG_LEN chars; rsplit may reduce that further.
            # Adding '…' means the final string is ≤MAX_ERROR_MSG_LEN + 1 chars.
            truncated = raw_msg[:MAX_ERROR_MSG_LEN]
            parts = truncated.rsplit(" ", 1)
            raw_msg = (parts[0] if len(parts) > 1 else truncated) + "…"
        error_msg = f"Interface navigation error: {raw_msg}"
        return {
            "audio": {
                "type": "error",
                "text": error_msg,
                "rate": 1.0,
                "interrupt": False,
            },
            "text": error_msg,
        }
