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
capped at 15 words.
"""

import numpy as np
from typing import Any, Dict, Optional

from model_router_client import copilot_llm_call

TOOL_NAME = "physical_interface_aid"

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
                "Identify the physical interface in view, list all visible "
                "buttons or controls with their labels and approximate positions, "
                "and note whether a user's finger or pointer is visible and where."
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are analyzing a physical interface for a blind user. "
                        "State: (1) the interface type (e.g. microwave keypad, "
                        "thermostat, kiosk, checkout terminal), (2) every visible "
                        "button or control with its label and grid position "
                        "(e.g. top-left, center-right), (3) whether a human finger "
                        "or stylus is visible and its approximate location on the "
                        "interface. Flag any obstructed or unclear elements. "
                        "If lighting is poor, say so. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "What kind of interface is this and where are all the "
                        "buttons? Is a finger visible?"
                    ),
                },
            ],
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "identify physical interface layout, buttons/controls, "
                    "and finger position"
                ),
            },
        )

        interface_artifact = interface_result.get("artifact") or {}
        interface_response = interface_result.get("response", "")

        # Use artifact if available; fall back to response text
        interface_context = (
            interface_artifact if interface_artifact else interface_response
        )

        # ── Stage 2: spatial reasoning — finger vs. buttons ──────────────────
        # Pass artifact only when it carries useful structured data; always
        # pass the original image so the model can inspect the scene directly.
        if interface_context:
            spatial_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are determining how a blind user's finger relates to "
                        "buttons on a physical interface. "
                        "Given the interface layout, state: "
                        "(1) which button the finger is currently touching or "
                        "nearest to (use the button's label), "
                        "(2) the exact direction the user needs to move to reach "
                        "the closest or most prominent button "
                        "(up, down, left, right, up-left, up-right, down-left, "
                        "down-right, or 'already there'), "
                        "(3) any nearby alternative buttons if direction is "
                        "ambiguous. If no finger is visible, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Interface layout: {interface_context}\n"
                        "Where is the finger relative to the buttons, and which "
                        "direction should the user move?"
                    ),
                },
            ]
        else:
            spatial_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are determining how a blind user's finger relates to "
                        "buttons on a physical interface visible in the image. "
                        "State: (1) which button the finger is nearest to, "
                        "(2) which direction to move to reach it, "
                        "(3) any ambiguous alternatives. "
                        "If no finger is visible, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Where is the finger relative to the interface buttons, "
                        "and which direction should the user move?"
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
        spatial_context = spatial_artifact if spatial_artifact else spatial_response

        # ── Stage 3: navigation — produce the spoken instruction ──────────────
        if spatial_context:
            nav_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are providing audio guidance for a blind user "
                        "navigating a physical interface. "
                        "Give ONE clear instruction in 15 words or fewer: "
                        "either confirm which button the finger is on "
                        "(e.g. 'Your finger is on the 7 button') "
                        "or give a brief direction to move "
                        "(e.g. 'Move right to the Start button'). "
                        "If direction is ambiguous, name the nearby buttons "
                        "(e.g. '5 is slightly left, 6 is slightly right'). "
                        "If the interface is unclear due to lighting or "
                        "obstruction, say so briefly. "
                        "Never exceed 15 words. No filler phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Interface: {interface_context}\n"
                        f"Spatial analysis: {spatial_context}\n"
                        "Give the spoken guidance now."
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
                        "Give ONE clear instruction in 15 words or fewer: "
                        "confirm which button the finger is on, or give a "
                        "direction to move. "
                        "Never exceed 15 words."
                    ),
                },
                {
                    "role": "user",
                    "content": "Guide the user to or confirm the nearest button.",
                },
            ]

        guidance_result = copilot_llm_call(
            capability="navigation",
            goal=(
                "Produce a brief spoken instruction (≤15 words) to guide the "
                "blind user's finger to or confirm the nearest interface button."
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

        # ── Streaming deduplication ───────────────────────────────────────────
        # Suppress repeat announcements when the scene hasn't changed
        if response == _last_response:
            return ""

        _last_response = response
        return response

    except Exception as exc:  # never raise — swallowed errors give no feedback
        raw_msg = str(exc)
        if len(raw_msg) > 150:
            # Truncate at a word boundary so the message stays readable
            raw_msg = raw_msg[:150].rsplit(" ", 1)[0] + "…"
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
