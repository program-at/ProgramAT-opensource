"""
Adaptive Visual Description Tool

Provides increasingly detailed descriptions of the visual scene based on how
much the camera is moving. Inspired by the WorldScribe system:

- Fast movement  → brief word-level object labels
- Slow movement  → general scene description with spatial relationships
- Stable/complex → detailed, proximity-prioritised descriptions

Streaming etiquette: returns "" when the scene or description level has not
changed to avoid repeating the same audio every frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Optional

from model_router_client import copilot_llm_call

TOOL_NAME = "adaptive_visual_description"

# Motion thresholds (mean absolute pixel difference, 0-255 scale)
MOTION_FAST_THRESHOLD = 30.0   # above → fast movement
MOTION_SLOW_THRESHOLD = 8.0    # above (and below fast) → slow movement
                                # at or below → stable

# Streaming: how many stable frames before upgrading to detailed description
STABLE_FRAMES_FOR_DETAIL = 4

# One description level is held for this many frames before re-querying
STREAMING_HOLD_FRAMES = 6

# Streaming word cap (per project convention)
STREAMING_WORD_LIMIT = 15

# Common everyday objects the user is likely to encounter; passed to the
# detector as a hint so it focuses on user-relevant targets rather than
# every COCO class.  This is intentionally a short, scene-agnostic list —
# the backend detector decides the final set of classes it evaluates.
FAST_MODE_TARGET_LABELS = [
    "person", "chair", "table", "car", "door", "window",
    "television", "refrigerator", "cabinet", "bed", "sofa",
    "laptop", "phone", "bottle", "cup", "bag", "book",
]

# ── Global streaming state ──────────────────────────────────────────────────
# Global variables follow the same per-tool pattern used by door_detection.py
# and other tools in this codebase.  Each tool runs in a single-user context
# on the backend server, so module-level state is safe and idiomatic here.
_prev_frame_gray: Optional[np.ndarray] = None   # previous frame for diff
_stable_frame_count: int = 0                    # consecutive stable frames
_last_description: str = ""                     # last spoken description
_last_level: str = ""                           # 'fast' | 'slow' | 'stable'
_hold_frame_count: int = 0                      # frames since last output
_consecutive_errors: int = 0                    # streaming failure counter

# Alert the user if this many consecutive streaming frames fail
_STREAMING_ERROR_THRESHOLD = 3


def _compute_motion(gray_current: np.ndarray, gray_prev: np.ndarray) -> float:
    """Return mean absolute difference between two grayscale frames."""
    diff = cv2.absdiff(gray_current, gray_prev)
    return float(np.mean(diff))


def _classify_motion(motion_score: float) -> str:
    """Classify motion score into 'fast', 'slow', or 'stable'."""
    if motion_score > MOTION_FAST_THRESHOLD:
        return "fast"
    if motion_score > MOTION_SLOW_THRESHOLD:
        return "slow"
    return "stable"


def _trim_to_word_limit(text: str, limit: int = STREAMING_WORD_LIMIT) -> str:
    """Trim text to at most `limit` words."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def _fast_description(image: np.ndarray) -> str:
    """Stage 1 – fast motion: return brief word-level object labels."""
    result = copilot_llm_call(
        capability="object_detection_localization",
        goal="List the most prominent objects visible as brief comma-separated labels.",
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "detect prominent objects and return brief comma-separated labels",
            "target_labels": FAST_MODE_TARGET_LABELS,
        },
    )
    artifact = result.get("artifact") or {}
    detections = artifact.get("detections") or []

    if detections:
        labels = []
        for det in detections:
            # The artifact schema varies by backend implementation; try the most
            # common key names in order of likelihood.
            label = det.get("label") or det.get("class") or det.get("name") or ""
            if label and label not in labels:
                labels.append(label.capitalize())
            if len(labels) >= 5:
                break
        if labels:
            return ", ".join(labels)

    # Fallback: use the model response directly
    response = (result.get("response") or "").strip()
    return _trim_to_word_limit(response) if response else ""


def _slow_description(image: np.ndarray) -> str:
    """Stage 1 – slow motion: general scene description with spatial layout."""
    result = copilot_llm_call(
        capability="general_reasoning",
        goal=(
            "Describe the scene in one sentence covering the main objects "
            "and their spatial relationships. Keep it under 15 words."
        ),
        messages=[
            {
                "role": "system",
                "content": "You are describing a scene for a blind user. Be concise and spatial.",
            },
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "generate concise scene description with spatial relationships",
        },
    )
    response = (result.get("response") or "").strip()
    return _trim_to_word_limit(response)


def _stable_description(image: np.ndarray) -> str:
    """Two-stage stable mode: spatial layout → detailed proximity description."""
    # Stage 1: spatial layout
    spatial_result = copilot_llm_call(
        capability="spatial_reasoning",
        goal=(
            "Identify visible objects and their positions relative to the viewer. "
            "Note which objects are closest."
        ),
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "identify object positions and proximity for blind user",
        },
    )
    spatial_artifact = spatial_result.get("artifact")

    # Stage 2: detailed description prioritised by proximity.
    # Build metadata separately so we only add the artifact key when useful.
    stage2_metadata: dict = {
        "tool_name": TOOL_NAME,
        "route_text": "detailed proximity-prioritised description for blind user",
    }
    if spatial_artifact:
        stage2_metadata["previous_stage_artifact"] = spatial_artifact

    detail_result = copilot_llm_call(
        capability="general_reasoning",
        goal=(
            "Give a detailed, audio-friendly description of the scene, "
            "starting with the nearest object. Keep it under 15 words."
        ),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are describing a scene for a blind user. "
                    "Prioritise objects by proximity. Be specific and concise."
                ),
            },
        ],
        images=[image],
        metadata=stage2_metadata,
    )
    response = (detail_result.get("response") or "").strip()
    return _trim_to_word_limit(response)


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """
    Adaptive Visual Description Tool entry point.

    Classifies camera motion and returns an appropriately detailed description:
      - Fast movement  → object labels only
      - Slow movement  → one-sentence scene description
      - Stable scene   → detailed, proximity-ordered description

    In streaming mode returns "" when nothing new to say.
    """
    global _prev_frame_gray, _stable_frame_count
    global _last_description, _last_level, _hold_frame_count, _consecutive_errors

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available."

    config = input_data if isinstance(input_data, dict) else {}
    is_streaming: bool = bool(config.get("is_streaming", False))

    # ── Compute motion ────────────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (160, 120))  # downscale for speed

    if _prev_frame_gray is None:
        _prev_frame_gray = gray_small
        if not is_streaming:
            # One-shot on first frame: treat as stable
            try:
                description = _stable_description(image)
                return description or "Scene not yet analysed."
            except Exception:  # noqa: BLE001
                return "Unable to analyze the scene. Try moving to a well-lit area and pointing the camera at the scene."
        return ""

    motion_score = _compute_motion(gray_small, _prev_frame_gray)
    _prev_frame_gray = gray_small
    level = _classify_motion(motion_score)

    # Track consecutive stable frames for upgrade to detailed description
    if level == "stable":
        _stable_frame_count += 1
    else:
        _stable_frame_count = 0

    # Promote to "stable" only after enough consecutive stable frames
    if level == "stable" and _stable_frame_count < STABLE_FRAMES_FOR_DETAIL:
        level = "slow"  # not yet stable enough for full detail

    # ── One-shot mode: always compute and return ──────────────────────────
    if not is_streaming:
        _last_description = ""
        _last_level = ""
        _hold_frame_count = 0
        _stable_frame_count = 0
        _consecutive_errors = 0

        try:
            if level == "fast":
                description = _fast_description(image)
            elif level == "slow":
                description = _slow_description(image)
            else:
                description = _stable_description(image)
            return description or "Unable to describe the scene."
        except Exception:  # noqa: BLE001
            return "Unable to analyze the scene. Try moving to a well-lit area and pointing the camera at the scene."

    # ── Streaming mode ────────────────────────────────────────────────────
    _hold_frame_count += 1

    # Don't re-query if we haven't held long enough AND level hasn't changed
    if level == _last_level and _hold_frame_count < STREAMING_HOLD_FRAMES:
        return ""

    # Query the appropriate description
    try:
        if level == "fast":
            description = _fast_description(image)
        elif level == "slow":
            description = _slow_description(image)
        else:
            description = _stable_description(image)
        _consecutive_errors = 0
    except Exception:  # noqa: BLE001
        _consecutive_errors += 1
        if _consecutive_errors >= _STREAMING_ERROR_THRESHOLD:
            _consecutive_errors = 0
            return "Scene analysis unavailable. Check camera and lighting."
        return ""  # Stay silent on isolated transient errors

    if not description:
        return ""

    # Suppress duplicate output
    if description == _last_description and level == _last_level:
        return ""

    _last_description = description
    _last_level = level
    _hold_frame_count = 0

    return description
