"""Provider-specific model adapters used by the shared policy executor."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import moondream_provider

from litellm_utils import call_model, call_openai_responses_model, extract_text


logger = logging.getLogger(__name__)
_STREAMING_VLM_DEBUG_LOCK = threading.Lock()
_STREAMING_VLM_DEBUG_SAVED = False
BACKEND_DIR = Path(__file__).resolve().parent
YOLO11_MODEL = "yolo11n.pt"
_IMPLEMENTATION_MODEL_CACHE: Dict[str, Any] = {}
class ModelAdapterError(ValueError):
    pass


class StreamingExecutionCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ImplementationProfile:
    name: str
    kind: str
    model: str = ""
    model_name: str = ""


@dataclass
class ImplementationResult:
    response: Any
    artifact: Any = None


ImplementationExecutor = Callable[
    [ImplementationProfile, List[Dict[str, Any]], Optional[Iterable[Any]], Dict[str, Any]],
    ImplementationResult,
]


def _response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        for key in ("text", "result", "answer", "content", "description"):
            if isinstance(response.get(key), str):
                return response[key].strip()
    try:
        return extract_text(response).strip()
    except Exception:
        return str(response).strip()


def _simple_response(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _model_executor(profile, messages, images, metadata) -> ImplementationResult:
    if not profile.model:
        raise ModelAdapterError(f"Model implementation {profile.name!r} has no model")
    tool_name = (metadata or {}).get("tool_name", "")
    has_images = bool(images)
    header = (
        f"[Model Prompt] model={profile.model}"
        f" tool={tool_name} has_images={has_images}"
    )
    print(header, flush=True)
    for msg in (messages or []):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = " ".join(text_parts)
        print(f"  [{role}] {content}", flush=True)
    logger.info(header)
    return ImplementationResult(call_model(profile.model, messages, images=images, metadata=metadata))


def _openai_responses_executor(profile, messages, images, metadata) -> ImplementationResult:
    """Call one configured OpenAI Responses API model; never choose fallbacks."""
    if not profile.model:
        raise ModelAdapterError(
            f"OpenAI Responses implementation {profile.name!r} has no model"
        )
    image_items = list(images or [])
    if not image_items:
        raise RuntimeError(f"OpenAI Responses implementation {profile.name!r} requires an image")
    prompt = next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(messages or [])
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        "",
    )
    text = call_openai_responses_model(
        profile.model,
        prompt,
        image_items[0],
        reasoning_effort=str(metadata.get("reasoning_effort") or "medium"),
    )
    return ImplementationResult(_simple_response(text), {"text": text})


def _image_bytes(image: Any) -> bytes:
    if isinstance(image, str):
        payload = image.split(",", 1)[1] if image.startswith("data:image") and "," in image else image
        return base64.b64decode(payload)
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    from PIL import Image
    import numpy as np
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, :3][:, :, ::-1].copy() if image.ndim == 3 else image)
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        return buffer.getvalue()
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def _preprocess_streaming_images(
    images: Iterable[Any], capability: str, original_payload_bytes: Optional[int] = None,
    original_image: Any = None,
) -> tuple[List[Any], float]:
    global _STREAMING_VLM_DEBUG_SAVED
    started = time.perf_counter()
    if capability == "ocr":
        max_dimension = int(os.environ.get("OCR_MAX_DIMENSION", "1024"))
        quality = int(os.environ.get("OCR_JPEG_QUALITY", "80"))
    else:
        max_dimension = int(os.environ.get("STREAMING_VLM_MAX_DIMENSION", "640"))
        quality = int(os.environ.get("STREAMING_VLM_JPEG_QUALITY", "70"))
    processed = []
    for image_index, image in enumerate(images):
        preprocess_source = original_image if image_index == 0 and original_image else image
        try:
            original_dimensions, original_bytes = moondream_provider.image_metrics(preprocess_source)
            payload, _mime_type, dimensions = moondream_provider.prepare_rgb_jpeg(
                preprocess_source, max_dimension, quality
            )
        except Exception as exc:
            logger.warning("[Streaming Image] preprocessing skipped error=%s", exc)
            processed.append(image)
            continue
        logged_original_bytes = (
            original_payload_bytes
            if image_index == 0 and original_payload_bytes is not None
            else original_bytes
        )
        logger.info(
            "[Streaming Image] original_dimensions=%sx%s original_kb=%.1f",
            original_dimensions[0], original_dimensions[1], logged_original_bytes / 1024,
        )
        logger.info(
            "[Streaming Image] vlm_dimensions=%sx%s vlm_payload_kb=%.1f quality=%d "
            "max_dimension=%d",
            dimensions[0], dimensions[1], len(payload) / 1024, quality, max_dimension,
        )
        if capability != "ocr" and not _STREAMING_VLM_DEBUG_SAVED:
            with _STREAMING_VLM_DEBUG_LOCK:
                if not _STREAMING_VLM_DEBUG_SAVED:
                    debug_path = Path(os.environ.get(
                        "MOONDREAM_DEBUG_IMAGE_PATH",
                        "/tmp/moondream_streaming_debug.jpg",
                    ))
                    try:
                        debug_path.parent.mkdir(parents=True, exist_ok=True)
                        debug_path.write_bytes(payload)
                        _STREAMING_VLM_DEBUG_SAVED = True
                        logger.info(
                            "[Streaming Image] corrected_debug_image=%s dimensions=%sx%s",
                            debug_path, dimensions[0], dimensions[1],
                        )
                    except OSError as exc:
                        logger.warning(
                            "[Streaming Image] debug image save failed path=%s error=%s",
                            debug_path, exc,
                        )
        processed.append(payload)
    return processed, (time.perf_counter() - started) * 1000


def _streaming_cancelled(metadata: Mapping[str, Any]) -> bool:
    check = metadata.get("_streaming_cancelled")
    return bool(callable(check) and check())


def _raise_if_streaming_cancelled(
    metadata: Mapping[str, Any], stage: str, streaming: bool
) -> None:
    if streaming and _streaming_cancelled(metadata):
        logger.info("[Streaming] cascade cancelled before=%s", stage)
        raise StreamingExecutionCancelled(f"Streaming stopped before {stage}")


MOONDREAM_BASE_URL = moondream_provider.BASE_URL
_MOONDREAM_STATE_LOCK = threading.Lock()
_MOONDREAM_COOLDOWN_UNTIL = 0.0
_MOONDREAM_CONSECUTIVE_FAILURES = 0


def _prepare_moondream_image(image: Any) -> tuple[bytes, str, tuple[int, int]]:
    """Return a Moondream-only RGB JPEG without altering other providers' input."""
    return moondream_provider.prepare_image(image)


def _build_moondream_messages(
    messages: Iterable[Mapping[str, Any]], image_data_uri: str
) -> List[Dict[str, Any]]:
    """Build the documented OpenAI content-list payload used in production."""
    return moondream_provider.build_messages(messages, image_data_uri)


def _create_moondream_client(api_key: str, timeout: float):
    return moondream_provider.create_client(api_key, timeout)


def _moondream_response_text(completion: Any) -> str:
    return moondream_provider.response_text(completion)


def _moondream_error_context(exc: Exception) -> tuple[Any, str, str]:
    return moondream_provider.error_context(exc)


def _moondream_cooldown_remaining() -> float:
    with _MOONDREAM_STATE_LOCK:
        return max(0.0, _MOONDREAM_COOLDOWN_UNTIL - time.monotonic())


def _record_moondream_success() -> None:
    global _MOONDREAM_CONSECUTIVE_FAILURES
    with _MOONDREAM_STATE_LOCK:
        _MOONDREAM_CONSECUTIVE_FAILURES = 0


def _record_moondream_failure(status: Any) -> tuple[float, int]:
    global _MOONDREAM_CONSECUTIVE_FAILURES, _MOONDREAM_COOLDOWN_UNTIL
    cooldown_seconds = max(
        0.0, float(os.environ.get("MOONDREAM_FAILURE_COOLDOWN_SECONDS", "60"))
    )
    threshold = max(1, int(os.environ.get("MOONDREAM_FAILURE_THRESHOLD", "3")))
    with _MOONDREAM_STATE_LOCK:
        _MOONDREAM_CONSECUTIVE_FAILURES += 1
        failures = _MOONDREAM_CONSECUTIVE_FAILURES
        if status == 500 or failures >= threshold:
            _MOONDREAM_COOLDOWN_UNTIL = time.monotonic() + cooldown_seconds
            _MOONDREAM_CONSECUTIVE_FAILURES = 0
            return cooldown_seconds, failures
    return 0.0, failures


def _moondream_cloud_executor(profile, messages, images, metadata) -> ImplementationResult:
    enabled = os.environ.get("ENABLE_MOONDREAM_CLOUD", "true").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Moondream Cloud is disabled")
    api_key = os.environ.get("MOONDREAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MOONDREAM_API_KEY is not configured")
    image_items = list(images or [])
    if not image_items:
        raise RuntimeError("Moondream Cloud requires an image")
    remaining = _moondream_cooldown_remaining()
    if remaining > 0:
        logger.info("[Moondream] cooldown active remaining_seconds=%.1f", remaining)
        raise RuntimeError("Moondream Cloud cooldown is active")

    model = os.environ.get("MOONDREAM_MODEL", "").strip() or profile.model
    timeout = float(os.environ.get("MOONDREAM_TIMEOUT_SECONDS", "2.0"))
    source_type = moondream_provider.input_type(image_items[0])
    if metadata.get("preserve_original_prompt"):
        prompt = moondream_provider.prompt_text(messages)
        short_task = prompt
        original_prompt_chars = len(prompt)
        prompt_source = "original_prompt_preserved"
    else:
        prompt, short_task, original_prompt_chars, prompt_source = moondream_provider.adapt_task_prompt(
            messages, metadata
        )
    card_mode, card_mode_reason = moondream_provider.card_mode_reason(short_task)
    payload, media_type, dimensions = _prepare_moondream_image(image_items[0])
    data_uri = f"data:{media_type};base64," + base64.b64encode(payload).decode("ascii")
    request_messages = _build_moondream_messages(
        [{"role": "user", "content": prompt}], data_uri
    )
    logger.info(
        "[Moondream] prompt_adapter=%s prompt_source=%s "
        "prompt_chars_original=%d prompt_chars_sent=%d",
        "none" if metadata.get("preserve_original_prompt") else "extractive_short",
        prompt_source, original_prompt_chars, len(prompt),
    )
    logger.info(
        "[Moondream] card_mode=%s reason=%s original_task_goal=%s",
        str(card_mode).lower(),
        json.dumps(card_mode_reason, ensure_ascii=False),
        json.dumps(short_task, ensure_ascii=False),
    )
    if card_mode:
        logger.info(
            "[Moondream] card_prompt_variant=%s",
            moondream_provider.CARD_PROMPT_VARIANT,
        )
    logger.info("[Moondream] prompt_preview=%r", prompt)
    logger.info(
        "[Moondream] request model=%s base_url=%s provider_input_type=%s "
        "converted_image=true dimensions=%sx%s payload_kb=%.1f prompt_chars=%d "
        "message_format=%s timeout_seconds=%.1f",
        model, MOONDREAM_BASE_URL, source_type,
        dimensions[0], dimensions[1], len(payload) / 1024, len(prompt),
        moondream_provider.MESSAGE_FORMAT, timeout,
    )
    logger.info("[Moondream] final_prompt_sent=%s", json.dumps(prompt, ensure_ascii=False))
    started = time.perf_counter()
    try:
        completion = _create_moondream_client(api_key, timeout).chat.completions.create(
            model=model,
            messages=request_messages,
        )
        raw_text = _moondream_response_text(completion)
        logger.info("[Moondream] raw_response=%s", json.dumps(raw_text, ensure_ascii=False))
        text = raw_text.strip()
        card_details = None
        if card_mode:
            # Card parsing remains diagnostic metadata only. Never replace the
            # model's generated text with the parsed/normalized representation.
            card_details = moondream_provider.card_response_details(raw_text)
            logger.info(
                "[Moondream] compact_notation_detected=%s",
                str(card_details["compact_notation_detected"]).lower(),
            )
            logger.info("[Moondream] extracted_cards=%s", json.dumps(
                card_details["extracted_cards"], ensure_ascii=False
            ))
            logger.info("[Moondream] deduped_cards=%s", json.dumps(
                card_details["deduped_cards"], ensure_ascii=False
            ))
            logger.info(
                "[Moondream] discarded_card_tokens=%s conflicting_suits=%s",
                json.dumps(card_details["discarded_card_tokens"], ensure_ascii=False),
                json.dumps(card_details["conflicting_suits"], ensure_ascii=False),
            )
            logger.info(
                "[Moondream] mirrored_corner_filter_applied=%s",
                str(card_details["mirrored_corner_filter_applied"]).lower(),
            )
            logger.info(
                "[Moondream] parsed_card_response=%s",
                json.dumps(card_details["final_response"], ensure_ascii=False),
            )
            logger.info(
                "[Moondream] card_result_valid=%s card_result_clean=%s",
                str(card_details["card_result_valid"]).lower(),
                str(card_details["card_result_clean"]).lower(),
            )
        if not text:
            raise RuntimeError("Moondream returned an empty or malformed response")
        logger.info("[Moondream] postprocess=none")
        _record_moondream_success()
        logger.info(
            "[Moondream] response status=200 request_id=%s latency_ms=%.1f text_chars=%d",
            getattr(completion, "_request_id", None) or "unavailable",
            (time.perf_counter() - started) * 1000,
            len(text),
        )
        artifact = {"text": text}
        if card_details is not None:
            artifact.update(card_details)
        return ImplementationResult(_simple_response(text), artifact)
    except Exception as exc:
        status, request_id, error_body = _moondream_error_context(exc)
        cooldown_seconds, failures = _record_moondream_failure(status)
        logger.warning(
            "[Moondream] failed status=%s request_id=%s error_type=%s error_body=%s",
            status, request_id, type(exc).__name__, error_body,
        )
        if cooldown_seconds:
            reason = "provider_backend_failure" if status == 500 else "repeated_failures"
            logger.warning(
                "[Moondream] temporarily disabled reason=%s cooldown_seconds=%.0f failures=%d",
                reason, cooldown_seconds, failures,
            )
        raise


def _location(bbox: Sequence[float], width: int, height: int) -> str:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    horizontal = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "center"
    vertical = "top" if y < height / 3 else "bottom" if y > 2 * height / 3 else "middle"
    return horizontal if vertical == "middle" else f"{vertical} {horizontal}"


def _target_labels(metadata: Mapping[str, Any]) -> List[str]:
    labels = metadata.get("target_labels") or metadata.get("targets") or []
    if isinstance(labels, str):
        labels = [labels]
    return [str(label).strip().lower() for label in labels if str(label).strip()]


def _object_detector_model(
    profile: ImplementationProfile, target_labels: Sequence[str]
) -> tuple[str, str]:
    logger.info(
        "[Target Grounding] detector=yolo model=%s target_labels=%s",
        profile.model_name or YOLO11_MODEL,
        list(target_labels),
    )
    return profile.model_name or YOLO11_MODEL, "yolo"


def _label_matches_targets(label: Any, target_labels: Sequence[str]) -> bool:
    normalized = str(label or "").strip().lower()
    return normalized in target_labels


def _yolo_executor(profile, messages, images, metadata) -> ImplementationResult:
    image_items = list(images or [])
    if not image_items:
        return ImplementationResult(_simple_response("No image was provided."), {"detections": []})
    from PIL import Image
    from ultralytics import YOLO
    image = Image.open(io.BytesIO(_image_bytes(image_items[0]))).convert("RGB")
    cache = metadata.get("model_cache") if isinstance(metadata.get("model_cache"), dict) else {}
    target_labels = _target_labels(metadata)
    model_name, _detector = _object_detector_model(profile, target_labels)
    model = cache.get(model_name) or YOLO(model_name)
    cache[model_name] = model
    detections = []
    for result in model(image, verbose=False):
        for box in result.boxes:
            bbox = [float(value) for value in box.xyxy[0].tolist()]
            detections.append({"label": str(result.names[int(box.cls[0])]), "bbox": bbox, "location": _location(bbox, image.width, image.height)})
    raw_labels = [item["label"] for item in detections]
    matching = [
        item for item in detections
        if not target_labels or _label_matches_targets(item["label"], target_labels)
    ]
    logger.info("[Target Grounding] YOLO raw_labels=%s kept_labels=%s", raw_labels, [item["label"] for item in matching])
    text = "; ".join(f"{item['label']} at {item['location']}" for item in matching)
    if not text:
        text = "No localized detection was produced for the requested object."
    return ImplementationResult(_simple_response(text), {
        "detections": matching,
        "target_labels": target_labels,
        "matching_detection": bool(matching),
        "accepted": bool(matching),
    })


IMPLEMENTATION_EXECUTORS: Dict[str, ImplementationExecutor] = {
    "model": _model_executor,
    "openai_responses": _openai_responses_executor,
    "moondream_cloud": _moondream_cloud_executor,
    "yolo": _yolo_executor,
}
