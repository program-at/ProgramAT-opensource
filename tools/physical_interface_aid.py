"""
Physical Interface Accessibility Aid

Helps blind users navigate physical interfaces (keypads, buttons, touchscreens,
thermostats, kiosks, checkout terminals, etc.) by providing real-time directional
guidance to locate and identify buttons.

Stages:
  1. structured_visual_understanding — identify the interface type and all
     visible buttons/controls, plus whether a finger/pointer is visible.
  2. spatial_reasoning — determine the finger's position relative to buttons
     and which direction the user should move.
  3. navigation — produce a short, spoken directional instruction.

Live mode: runs every ~1 second; returns "" when nothing has changed so
the same phrase is not repeated continuously. Streaming responses are
capped at 15 words via both system-prompt instruction and code enforcement.
Output format (cardinal-only directions, no banned verbs) is also enforced
in code so restrictions apply regardless of whether the backend's planning
and routing pipeline is enabled.
"""

import re
import numpy as np
from typing import Any, Dict, Optional

from model_router_client import copilot_llm_call

TOOL_NAME = "physical_interface_aid"

# Maximum words in a streaming/live-mode response (enforced in code + via prompt)
STREAMING_WORD_LIMIT = 15

# Maximum characters for the raw error message portion (truncated at word boundary)
MAX_ERROR_MSG_LEN = 150

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

    Applied after the Stage 3 LLM response is received:
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


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Main entry point for the physical interface accessibility aid.

    Args:
        image:      Camera frame as a numpy array (BGR, from OpenCV). May be None.
        input_data: Optional config dict (currently unused; reserved for future
                    options such as a target button name).

    Returns:
        A spoken string (≤ 15 words in streaming/live mode) directing the user
        to or confirming the button their finger is on.  Returns "" when the
        scene has not changed, so TTS is not triggered redundantly.
        Returns an audio-error dict when no image is available.
    """
    global _last_response, _frame_count
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
        # ── Stage 1: understand the interface layout ─────────────────────────
        interface_result = copilot_llm_call(
            capability="structured_visual_understanding",
            goal=(
                "Identify the physical interface in view, list all interactive "
                "buttons or controls with their labels and approximate positions, "
                "and note whether a user's finger or pointer is visible and where. "
                "Do NOT list displays, screens, digital readouts, status indicators, "
                "or any non-interactive elements."
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are analyzing a physical interface for a blind user. "
                        "State: (1) the interface type (e.g. microwave keypad, "
                        "thermostat, kiosk, checkout terminal), (2) every interactive "
                        "button or control with its label and grid position "
                        "(e.g. top-left, center-right) — do NOT include displays, "
                        "screens, digital readouts, clocks, timers, or status "
                        "indicators; list only elements the user can physically "
                        "activate, (3) whether a human finger or stylus is visible "
                        "and its approximate location on the interface. "
                        "Flag any obstructed or unclear buttons. "
                        "If lighting is poor, say so. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "What kind of interface is this and where are all the "
                        "interactive buttons? Is a finger visible?"
                    ),
                },
            ],
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "identify physical interface interactive buttons/controls "
                    "and finger position"
                ),
            },
        )

        interface_artifact = interface_result.get("artifact") or {}
        interface_response = interface_result.get("response", "")

        # Use artifact when present; fall back to the text response.
        # Both empty dict and empty string are falsy, so `or` handles both.
        interface_context = interface_artifact or interface_response

        # ── Stage 2: spatial reasoning — finger vs. buttons ──────────────────
        # Pass artifact only when it carries useful structured data; always
        # pass the original image so the model can inspect the scene directly.
        if interface_context:
            spatial_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are determining how a blind user's finger relates to "
                        "the nearest interactive button on a physical interface. "
                        "Focus only on the single button closest to the finger — "
                        "do NOT describe other buttons, display elements, or any "
                        "part of the interface not relevant to the finger's position. "
                        "State: "
                        "(1) Is the finger DIRECTLY ON that button — meaning it "
                        "clearly overlaps the button's area? Or is it only NEARBY "
                        "or ADJACENT without overlapping? Use the button's label "
                        "and state 'on' or 'near'. "
                        "(2) If the finger is NOT directly on the button, give the "
                        "exact direction needed to move onto it "
                        "(up, down, left, right, up-left, up-right, down-left, "
                        "down-right). Only use 'already there' if the finger "
                        "clearly and unambiguously overlaps the button. "
                        "(3) Only if two buttons are genuinely equidistant, name "
                        "both. If no finger is visible, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Interface layout: {interface_context}\n"
                        "Which single button is the finger closest to? Is the "
                        "finger directly on it (overlapping) or only near it? "
                        "If near, which direction must the user move?"
                    ),
                },
            ]
        else:
            spatial_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are determining how a blind user's finger relates to "
                        "the nearest interactive button on a physical interface "
                        "visible in the image. "
                        "Focus only on the single button closest to the finger — "
                        "do NOT describe other buttons, displays, or unrelated "
                        "elements. "
                        "State: (1) Is the finger DIRECTLY ON that button "
                        "(overlapping its area) or only NEAR/ADJACENT? Use 'on' "
                        "or 'near'. "
                        "(2) If the finger is NOT directly on the button, give the "
                        "direction needed to move onto it. Only use 'already there' "
                        "if the finger clearly overlaps the button. "
                        "(3) Only if two buttons are genuinely equidistant, name both. "
                        "If no finger is visible, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Which single button is the finger closest to? Is the "
                        "finger directly on it or only near it? "
                        "If near, which direction should the user move?"
                    ),
                },
            ]

        spatial_result = copilot_llm_call(
            capability="spatial_reasoning",
            goal=(
                "Determine the relationship between the user's finger and the "
                "nearest interface button, and the direction needed to reach it."
            ),
            messages=spatial_messages,
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "determine finger position relative to physical interface buttons"
                ),
                "previous_stage_artifact": interface_artifact or None,
            },
        )

        spatial_artifact = spatial_result.get("artifact") or {}
        spatial_response = spatial_result.get("response", "")
        # Both empty dict and empty string are falsy.
        spatial_context = spatial_artifact or spatial_response

        # ── Stage 3: navigation — produce the spoken instruction ──────────────
        if spatial_context:
            nav_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are providing audio guidance for a blind user "
                        "navigating a physical interface. "
                        "The user cannot see. "
                        "Only reference the single button or element that is "
                        "relevant to the user's finger — do NOT mention other "
                        "buttons, display elements, status readouts, or any part "
                        "of the interface not immediately relevant to where the "
                        "finger is or needs to go. "
                        "You MUST output EXACTLY ONE of these three forms — nothing else: "
                        "  A) 'your finger is on [element]' — only when the "
                        "spatial analysis explicitly states the finger is ON or directly "
                        "overlapping that element. "
                        "  B) 'move [direction] towards [element]' — when the finger "
                        "is near but NOT on an element. Direction must be exactly one "
                        "word: left, right, up, or down. "
                        "  C) '[element a] is slightly [direction] of your finger, "
                        "[element b] is slightly [opposite direction] of your finger' — "
                        "only when two elements are equidistant and direction is "
                        "genuinely ambiguous. "
                        "NEVER use: touch, touching, tap, tapping, press, pressing, "
                        "reach, reaching, find, finding, locate, locating. "
                        "NEVER use diagonal directions. "
                        "NEVER use vague instructions. NEVER exceed 15 words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Spatial analysis: {spatial_context}\n"
                        "Give the spoken guidance now — reference only the element "
                        "the finger is on or needs to reach."
                    ),
                },
            ]
        else:
            nav_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are providing audio guidance for a blind user "
                        "navigating a physical interface visible in the image. "
                        "The user cannot see. "
                        "Only reference the single button or element that is "
                        "relevant to the user's finger — do NOT mention other "
                        "buttons, display elements, status readouts, or any part "
                        "of the interface not immediately relevant to where the "
                        "finger is or needs to go. "
                        "You MUST output EXACTLY ONE of these three forms — nothing else: "
                        "  A) 'your finger is on [element]' — only when the "
                        "finger clearly overlaps the element area. "
                        "  B) 'move [direction] towards [element]' — when the finger "
                        "is near but not on an element. Direction must be exactly one "
                        "word: left, right, up, or down. "
                        "  C) '[element a] is slightly [direction] of your finger, "
                        "[element b] is slightly [opposite direction] of your finger' — "
                        "only when two elements are equidistant and direction is "
                        "genuinely ambiguous. "
                        "NEVER use: touch, touching, tap, tapping, press, pressing, "
                        "reach, reaching, find, finding, locate, locating. "
                        "NEVER use diagonal directions. "
                        "NEVER use vague instructions. NEVER exceed 15 words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Give the spoken guidance now — reference only the element "
                        "the finger is on or needs to reach."
                    ),
                },
            ]

        guidance_result = copilot_llm_call(
            capability="navigation",
            goal=(
                "Produce a spoken instruction (≤15 words) referencing only the "
                "single button relevant to the user's finger position. "
                "Use one of three strict forms: "
                "'your finger is on [element]' when the finger overlaps the element; "
                "'move [direction] towards [element]' with a single cardinal direction "
                "(left, right, up, or down only — no diagonals); or "
                "'[element a] is slightly [direction] of your finger, [element b] is "
                "slightly [opposite direction] of your finger' when equidistant. "
                "Never mention displays, readouts, or unrelated interface elements. "
                "Never use: touch, touching, tap, tapping, press, pressing, reach, reaching, "
                "find, finding, locate, locating."
            ),
            messages=nav_messages,
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "generate directional guidance for physical interface navigation"
                ),
                "previous_stage_artifact": spatial_artifact or None,
            },
        )

        response: str = guidance_result.get("response", "").strip()

        if not response:
            return ""

        # ── Enforce 15-word cap (safety net on top of prompt instruction) ────
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

        _last_response = response
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
