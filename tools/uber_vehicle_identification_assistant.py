"""Uber vehicle identification assistant for blind and low-vision users."""

from __future__ import annotations

from typing import Any, Dict

from model_router_client import copilot_llm_call

TOOL_NAME = "uber_vehicle_identification_assistant"
TARGET_LABELS = ["car", "vehicle", "license plate", "suv", "sedan"]
STREAMING_WORD_LIMIT = 15


def _as_dict(input_data: Any) -> Dict[str, Any]:
    return input_data if isinstance(input_data, dict) else {}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _extract_expected_vehicle(input_data: Dict[str, Any]) -> Dict[str, str]:
    expected = input_data.get("expected_vehicle")
    if not isinstance(expected, dict):
        expected = {}

    make = _clean_text(
        input_data.get("make")
        or input_data.get("target_make")
        or input_data.get("expected_make")
        or expected.get("make")
    )
    model = _clean_text(
        input_data.get("model")
        or input_data.get("target_model")
        or input_data.get("expected_model")
        or expected.get("model")
    )
    color = _clean_text(
        input_data.get("color")
        or input_data.get("target_color")
        or input_data.get("expected_color")
        or expected.get("color")
    )
    plate = _clean_text(
        input_data.get("plate")
        or input_data.get("license_plate")
        or input_data.get("target_plate")
        or input_data.get("expected_plate")
        or expected.get("plate")
    )

    result = {"make": make, "model": model, "color": color, "plate": plate}
    return {key: value for key, value in result.items() if value}


def _criteria_text(criteria: Dict[str, str], fallback_query: str) -> str:
    if criteria:
        parts = [f"{key}: {value}" for key, value in criteria.items()]
        return "User-provided criteria: " + ", ".join(parts) + "."
    if fallback_query:
        return f"User request: {fallback_query}"
    return "No specific expected vehicle criteria were provided."


def _artifact_usable(stage_result: Any) -> bool:
    if not isinstance(stage_result, dict):
        return False

    artifact = stage_result.get("artifact")
    if artifact in (None, "", [], {}):
        return False

    if isinstance(artifact, dict):
        confidence = artifact.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < 0.35:
            return False
        if "text" in artifact and not _clean_text(artifact.get("text")):
            return False
        detections = artifact.get("detections")
        if isinstance(detections, list) and not detections:
            return False
    return True


def _truncate_for_streaming(text: str, streaming: bool) -> str:
    words = text.split()
    cleaned = " ".join(words)
    if not streaming or len(words) <= STREAMING_WORD_LIMIT:
        return cleaned
    return " ".join(words[:STREAMING_WORD_LIMIT])


def _has_image_data(image: Any) -> bool:
    if image is None:
        return False
    ndim = getattr(image, "ndim", None)
    if isinstance(ndim, int) and ndim < 2:
        return False
    size = getattr(image, "size", None)
    if isinstance(size, int):
        return size > 0
    if isinstance(size, tuple):
        return all(isinstance(value, int) and value > 0 for value in size)
    return bool(size)


def main(image: Any, input_data: Any = None) -> Any:
    if not _has_image_data(image):
        return {
            "audio": {"type": "error", "text": "No camera image available."},
            "text": "No camera image available.",
        }

    config = _as_dict(input_data)
    expected_vehicle = _extract_expected_vehicle(config)
    user_query = _clean_text(config.get("query") or config.get("prompt"))
    criteria_text = _criteria_text(expected_vehicle, user_query)
    is_streaming = bool(
        config.get("streaming")
        or config.get("is_streaming")
        or config.get("live_mode")
    )

    detection = copilot_llm_call(
        capability="object_detection_localization",
        goal="Find candidate vehicles near the center and locate visible license plates.",
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "locate cars and likely license-plate regions for Uber matching",
            "target_labels": TARGET_LABELS,
        },
    )

    ocr_metadata: Dict[str, Any] = {
        "tool_name": TOOL_NAME,
        "route_text": "read plate text from likely target vehicle",
    }
    if _artifact_usable(detection):
        ocr_metadata["previous_stage_artifact"] = detection.get("artifact")

    ocr = copilot_llm_call(
        capability="ocr",
        goal="Extract license plate text for the most likely target vehicle.",
        messages=[{"role": "user", "content": "Prioritize plate text nearest the center vehicle."}],
        images=[image],
        metadata=ocr_metadata,
    )

    reasoning_metadata: Dict[str, Any] = {
        "tool_name": TOOL_NAME,
        "route_text": "identify make/model/color/plate and estimate if this is the user's Uber",
    }
    usable_artifacts: Dict[str, Any] = {}
    if _artifact_usable(detection):
        usable_artifacts["vehicle_detection"] = detection.get("artifact")
    if _artifact_usable(ocr):
        usable_artifacts["ocr"] = ocr.get("artifact")
    if usable_artifacts:
        reasoning_metadata["previous_stage_artifact"] = usable_artifacts

    reasoning = copilot_llm_call(
        capability="general_reasoning",
        goal="Identify vehicle details and compare against user criteria.",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are assisting a blind user. "
                    "Return concise, audio-friendly plain text. "
                    "Include make, model, color, and plate if visible. "
                    "Then state whether it likely matches the user's Uber criteria."
                ),
            },
            {
                "role": "user",
                "content": criteria_text,
            },
        ],
        images=[image],
        metadata=reasoning_metadata,
    )

    response = _clean_text(reasoning.get("response")) if isinstance(reasoning, dict) else ""
    if not response:
        return "I could not confirm enough vehicle details yet. Please point the camera at the car."
    return _truncate_for_streaming(response, is_streaming)


run = main
process_image = main
