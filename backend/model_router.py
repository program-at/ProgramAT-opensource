"""Atomic capability execution and fixed infrastructure LLM calls for ProgramAT."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
import urllib.error
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml
import moondream_provider

from litellm_utils import call_model, extract_text


logger = logging.getLogger(__name__)
_STREAMING_VLM_DEBUG_LOCK = threading.Lock()
_STREAMING_VLM_DEBUG_SAVED = False
BACKEND_DIR = Path(__file__).resolve().parent
CAPABILITY_PROFILES_PATH = BACKEND_DIR / "capability_profiles.yaml"
EXECUTION_POLICY_PATH = BACKEND_DIR / "execution_policy.yaml"
DEFAULT_FALLBACK_CAPABILITY = "general_reasoning"
LEGACY_CAPABILITY_ALIASES = {
    "object_detection": "object_detection_localization",
    "spatial_relationship": "spatial_reasoning",
    "map_web": "structured_visual_understanding",
    "video": "temporal_reasoning",
}
NAVIGATION_SYSTEM_PROMPT = (
    "You provide navigation guidance for a blind or low-vision user. "
    "Use useful previous-stage detection results as additional context. "
    "A missing, failed, or uncertain detection is not evidence that the target is absent; "
    "inspect the image independently when prior-stage context is uncertain. "
    "Give concise spoken instructions "
    "in at most 2 short sentences. Do not use a numbered list, introduction, "
    "explanation, conversational filler, safety disclaimer, 'keep an eye out', "
    "or 'don't hesitate to ask'."
)
AUDIO_RESPONSE_PROMPT = (
    "Return only concise, audio-friendly plain text in one line, preferably 1-2 short "
    "sentences. Do not use Markdown, headings, bullet points, numbered lists, or newline "
    "characters. For ordered information, use commas or semicolons instead of line breaks."
)
EVALUATION_PROMPT = """Judge only whether the candidate response is sufficiently useful, informative, accessible, and actionable for the user's request. You cannot see the image. Do not fact-check visual claims or reject because you are unsure whether a visual claim is true.

Output YES if the response:
- provides actionable useful information for at least part of the task;
- clearly states uncertainty for parts it cannot determine;
- helps the user make progress even if it does not fully solve every item;
- gives a reasonable next action when something remains unclear;
- is specific and actionable for a blind or low-vision user;
- uses visual descriptors appropriately or supplements them with actionable cues.

Output NO if the response:
- gives no useful actionable information or fails to address the task;
- is too vague or generic to help;
- fails to distinguish multiple physical items using usable cues;
- relies mainly on inaccessible visual cues such as color when color was not requested;
- gives partial information in a confusing way that could make the user act on the wrong item.

Do not ban color or appearance. It is acceptable when requested, supplementary, or paired with position, order, size, proximity, or a next action. Prefer cues such as left/right, top/bottom, first/second, closest/farthest, next to, open/recycle, move closer, or turn.

Partial but useful answers should usually be accepted, especially for streaming assistive tools. Do not output NO merely because the answer is incomplete, not every visible item is categorized, uncertainty is clearly communicated, or another model might provide more detail.

Mail examples:
- NO: "The blue envelope is junk and the white envelope is important."
- YES: "The bottom mailer is a junk credit card offer; the top envelope addressed to you is likely important."
- YES: "This appears to be a promotional flyer for a dental office, which is junk mail. There is also a partial view of an envelope at the top of the table, but I cannot read the details on it."

Be conservative but not overly strict. Do not reject merely because wording could improve or minor details could be added.

Capability/task type: {capability}
Original user request or tool goal: {goal}
Previous-stage textual outputs, if any: {previous_text}
Candidate response: {response}

Output exactly one token: YES or NO."""
EVALUATION_REASON_PROMPT = """Explain in one short sentence why the evaluator decision below passed or failed on usefulness, actionability, accessibility, partial progress, and clearly stated uncertainty. Do not inspect or fact-check an image.

Capability/task type: {capability}
Original user request or tool goal: {goal}
Previous-stage textual outputs, if any: {previous_text}
Candidate response: {response}
Decision: {decision}

Output only the concise reason sentence."""
STREAMING_RESPONSE_PROMPT = (
    "You are responding to a live camera stream. Answer in 1-2 short, audio-friendly "
    "sentences. Prioritize only the most useful information for the user right now. "
    "Do not give long bullet lists, full reports, or repeated reasoning. For mail "
    "categorization, state the likely category and one brief reason. When multiple "
    "physical items are present, do not rely on color alone; prefer concise actionable "
    "cues such as position, order, size, proximity, or the next action. Include extra "
    "detail only when important for action or safety."
)
TARGET_LABEL_ALIASES = {
    "exit": {"exit", "door", "doorway", "exit sign"},
    "door": {"exit", "door", "doorway", "exit sign"},
    "doorway": {"exit", "door", "doorway", "exit sign"},
    "exit sign": {"exit", "door", "doorway", "exit sign"},
}
EXIT_TARGET_LABELS = ["exit", "door", "doorway", "exit sign"]
COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}
YOLO11_MODEL = "yolo11n.pt"
YOLOWORLD_MODEL = "yolov8s-world.pt"
_IMPLEMENTATION_MODEL_CACHE: Dict[str, Any] = {}
STREAMING_EXECUTION_CONTEXT: ContextVar[bool] = ContextVar(
    "streaming_execution", default=False
)
TOOL_EXECUTION_IMAGES: ContextVar[Optional[List[Any]]] = ContextVar(
    "tool_execution_images", default=None
)


class ExecutionPolicyError(ValueError):
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


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def normalize_capability_name(capability: Any) -> str:
    """Return the canonical capability name accepted by execution policies."""
    name = str(capability or "").strip().lower()
    return LEGACY_CAPABILITY_ALIASES.get(name, name)


def load_capability_profiles(path: Path = CAPABILITY_PROFILES_PATH) -> Dict[str, Dict[str, Any]]:
    capabilities = _load_yaml(path).get("capabilities", {})
    if not isinstance(capabilities, dict) or not capabilities:
        raise ValueError(f"No capabilities configured in {path}")
    profiles: Dict[str, Dict[str, Any]] = {}
    for capability, raw in capabilities.items():
        name = str(capability).strip()
        if isinstance(raw, list):
            values = [str(value).strip() for value in raw if str(value).strip()]
            profiles[name] = {
                "description": values[0] if values else "",
                "include_examples": values[1:],
                "exclude_examples": [],
                "notes": "",
            }
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Capability {name!r} must be a list or mapping")
        include_examples = raw.get("include_examples", []) or []
        exclude_examples = raw.get("exclude_examples", []) or []
        if not isinstance(include_examples, list) or not isinstance(exclude_examples, list):
            raise ValueError(f"Capability {name!r} examples must be lists")
        profiles[name] = {
            "description": str(raw.get("description", "")).strip(),
            "include_examples": [str(value).strip() for value in include_examples if str(value).strip()],
            "exclude_examples": [str(value).strip() for value in exclude_examples if str(value).strip()],
            "notes": str(raw.get("notes", "")).strip(),
        }
    return profiles


def load_capability_descriptions(path: Path = CAPABILITY_PROFILES_PATH) -> Dict[str, List[str]]:
    descriptions = {}
    for capability, profile in load_capability_profiles(path).items():
        values = [profile["description"], *profile["include_examples"]]
        if profile["notes"]:
            values.append(profile["notes"])
        cleaned = [value for value in values if value]
        if not cleaned:
            raise ValueError(f"Capability {capability!r} must have a description or example")
        descriptions[capability] = cleaned
    return descriptions


def load_execution_policies(path: Path = EXECUTION_POLICY_PATH) -> Dict[str, Dict[str, Any]]:
    config = _load_yaml(path)
    config.pop("global", None)
    config.pop("implementations", None)
    config.pop("streaming", None)
    cascade_profiles = config.pop("cascade_profiles", {})
    if not isinstance(cascade_profiles, dict):
        raise ExecutionPolicyError("cascade_profiles must be a mapping")

    policies = {}
    for capability, raw in config.items():
        if not isinstance(raw, dict):
            raise ExecutionPolicyError(f"Capability {capability!r} must be a mapping")
        implementation = str(raw.get("implementation") or "").strip()
        cascade_name = str(raw.get("cascade") or "").strip()
        if bool(implementation) == bool(cascade_name):
            raise ExecutionPolicyError(
                f"Capability {capability!r} must define exactly one of implementation or cascade"
            )
        if implementation:
            policy = {
                "candidates": [implementation],
                "evaluator": None,
                "cascade": None,
                "specialized": bool(raw.get("specialized", False)),
            }
        else:
            profile = cascade_profiles.get(cascade_name)
            if not isinstance(profile, dict):
                raise ExecutionPolicyError(
                    f"Capability {capability!r} references unknown cascade profile {cascade_name!r}"
                )
            candidates = [
                str(value).strip()
                for value in profile.get("candidates", [])
                if str(value).strip()
            ]
            evaluator = str(profile.get("evaluator") or "").strip()
            if not candidates or not evaluator:
                raise ExecutionPolicyError(
                    f"Cascade profile {cascade_name!r} requires candidates and evaluator"
                )
            policy = {
                "candidates": candidates,
                "evaluator": evaluator,
                "cascade": cascade_name,
                "specialized": bool(raw.get("specialized", False)),
            }
        policies[str(capability).strip()] = policy
    taxonomy = set(load_capability_profiles())
    if set(policies) != taxonomy:
        missing = taxonomy - set(policies)
        extra = set(policies) - taxonomy
        raise ExecutionPolicyError(
            "Capability taxonomy mismatch between capability_profiles.yaml and "
            "execution_policy.yaml; "
            f"missing_policies={sorted(missing)}, unknown_policies={sorted(extra)}"
        )
    return policies


def load_implementation_profiles(path: Path = EXECUTION_POLICY_PATH) -> Dict[str, ImplementationProfile]:
    raw_profiles = _load_yaml(path).get("implementations", {})
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ExecutionPolicyError(f"No implementations configured in {path}")
    profiles = {}
    for name, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ExecutionPolicyError(f"Implementation {name!r} must be a mapping")
        profiles[str(name)] = ImplementationProfile(
            name=str(name),
            kind=str(raw.get("kind", "model")).strip(),
            model=str(raw.get("model", "")).strip(),
            model_name=str(raw.get("model_name", "")).strip(),
        )
    return profiles


def load_global_execution_config(path: Path = EXECUTION_POLICY_PATH) -> Dict[str, Any]:
    raw = _load_yaml(path).get("global", {})
    if not isinstance(raw, dict):
        raise ExecutionPolicyError("global execution policy must be a mapping")
    planner_enabled = raw.get("planner_enabled")
    routing_enabled = raw.get("routing_enabled")
    if not isinstance(planner_enabled, bool):
        raise ExecutionPolicyError("global.planner_enabled must be true or false")
    if not isinstance(routing_enabled, bool):
        raise ExecutionPolicyError("global.routing_enabled must be true or false")
    if not planner_enabled and routing_enabled:
        logger.warning(
            "[Execution Policy] planner_enabled=false with routing_enabled=true is invalid; "
            "forcing routing_enabled=false"
        )
        routing_enabled = False

    def implementation_name(field: str) -> str:
        value = raw.get(field)
        if not isinstance(value, dict) or not str(value.get("implementation") or "").strip():
            raise ExecutionPolicyError(f"global.{field}.implementation is required")
        return str(value["implementation"]).strip()

    return {
        "planner_enabled": planner_enabled,
        "routing_enabled": routing_enabled,
        "system_model": implementation_name("system_model"),
        "default_llm_when_routing_disabled": implementation_name(
            "default_llm_when_routing_disabled"
        ),
    }


def _log_global_execution_config(
    config: Mapping[str, Any],
    implementations: Mapping[str, ImplementationProfile],
) -> None:
    system_name = config["system_model"]
    default_name = config["default_llm_when_routing_disabled"]
    system_profile = implementations.get(system_name)
    default_profile = implementations.get(default_name)
    logger.info(
        "[Execution Policy] planner_enabled=%s routing_enabled=%s",
        str(config["planner_enabled"]).lower(),
        str(config["routing_enabled"]).lower(),
    )
    logger.info(
        "[Execution Policy] system_model=%s/%s",
        system_name,
        system_profile.model if system_profile else "<unknown>",
    )
    logger.info(
        "[Execution Policy] default_llm_when_routing_disabled=%s/%s",
        default_name,
        default_profile.model if default_profile else "<unknown>",
    )


def validate_execution_configuration(
    policies: Optional[Mapping[str, Mapping[str, Any]]] = None,
    implementations: Optional[Mapping[str, ImplementationProfile]] = None,
) -> None:
    taxonomy = set(load_capability_profiles())
    policies = policies or load_execution_policies()
    implementations = implementations or load_implementation_profiles()
    global_config = load_global_execution_config()

    policy_names = set(policies)
    if policy_names != taxonomy:
        missing = taxonomy - policy_names
        extra = policy_names - taxonomy
        raise ExecutionPolicyError(
            "Capability taxonomy mismatch between capability_profiles.yaml and "
            "execution_policy.yaml; "
            f"missing_policies={sorted(missing)}, unknown_policies={sorted(extra)}"
        )

    configured_names = {
        name
        for policy in policies.values()
        for name in [*policy["candidates"], policy.get("evaluator")]
        if name
    }
    unknown = configured_names - set(implementations)
    if unknown:
        raise ExecutionPolicyError(f"Unknown implementations: {', '.join(sorted(unknown))}")
    global_names = {
        global_config["system_model"],
        global_config["default_llm_when_routing_disabled"],
    }
    unknown_global = global_names - set(implementations)
    if unknown_global:
        raise ExecutionPolicyError(
            f"Unknown global implementations: {', '.join(sorted(unknown_global))}"
        )


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


def _single_line_audio_text(text: Any) -> str:
    """Normalize generated user-facing text into plain, single-line speech."""
    lines = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        lines.append(" ".join(line.split()))
    result = ""
    for line in lines:
        if not result:
            result = line
        elif result.endswith(":"):
            result += " " + line
        else:
            result += "; " + line
    return result


try:
    validate_execution_configuration()
except Exception as exc:
    raise ExecutionPolicyError(f"Execution policy configuration invalid: {exc}") from exc


def _simple_response(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _model_executor(profile, messages, images, metadata) -> ImplementationResult:
    if not profile.model:
        raise ExecutionPolicyError(f"Model implementation {profile.name!r} has no model")
    return ImplementationResult(call_model(profile.model, messages, images=images, metadata=metadata))


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


_CARD_RANK_PATTERN = re.compile(
    r"\b(?:ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|[2-9]|10)\b",
    re.IGNORECASE,
)
_CARD_SUIT_PATTERN = re.compile(r"\b(?:spades?|hearts?|diamonds?|clubs?)\b", re.IGNORECASE)
_STREAMING_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:cannot|can't|unable|not sure|uncertain|unclear|could not|couldn't|cannot determine)\b",
    re.IGNORECASE,
)
_SINGLE_CARD_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:cannot|can't|unable|not sure|uncertain|unclear|could not|couldn't|"
    r"cannot determine|cannot tell|can't tell|possibly|maybe|might be|"
    r"appears to be|looks like|likely)\b",
    re.IGNORECASE,
)
_CARD_COUNT_WORDS = {
    "one": 1,
    "a": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "multiple": 2,
    "several": 2,
}
_CARD_COUNT_PATTERN = re.compile(
    r"\b(?P<count>one|a|single|two|three|four|five|six|seven|eight|nine|ten|multiple|several|\d+)\s+"
    r"(?:visible\s+)?(?:playing\s+)?cards?\b",
    re.IGNORECASE,
)
_CARD_LIST_TERMS = ("cards", "visible cards", "playing cards")


def _parse_json_like(value: Any) -> Optional[Any]:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (TypeError, ValueError):
        return None


def _card_count_from_text(text: str) -> Optional[int]:
    counts = []
    for match in _CARD_COUNT_PATTERN.finditer(text):
        raw_count = match.group("count").lower()
        count = int(raw_count) if raw_count.isdigit() else _CARD_COUNT_WORDS.get(raw_count)
        if count is not None:
            counts.append(count)
    if not counts:
        return None
    return counts[0] if len(set(counts)) == 1 else -1


def _count_cards_in_json(value: Any) -> Optional[int]:
    parsed = _parse_json_like(value)
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return len(parsed)
    if not isinstance(parsed, dict):
        return None
    for key in ("cards", "visible_cards", "playing_cards"):
        cards = parsed.get(key)
        if isinstance(cards, list):
            return len(cards)
    for key in ("card_count", "count", "visible_card_count"):
        raw_count = parsed.get(key)
        if isinstance(raw_count, int):
            return raw_count
        if isinstance(raw_count, str) and raw_count.strip().isdigit():
            return int(raw_count.strip())
    if parsed.get("rank") and parsed.get("suit"):
        return 1
    return None


def _is_single_card_task(tool_name: Any, task_type: Any) -> bool:
    normalized_tool = str(tool_name or "").strip().lower()
    normalized_task = normalize_capability_name(str(task_type or "").strip().lower())
    return (
        normalized_task in {
            "visual_representation_understanding",
            "structured_visual_understanding",
        }
        or normalized_tool == "playing_card_reader"
    )


def _single_card_evaluator_skip_reason(
    result: Any, tool_name: Any, task_type: Any
) -> tuple[bool, str]:
    if not _is_single_card_task(tool_name, task_type):
        return False, "not visual card reader task"
    text = " ".join(str(result or "").split())
    if not text:
        return False, "empty main model output"
    if _SINGLE_CARD_UNCERTAINTY_PATTERN.search(text):
        return False, "uncertain main model output"
    normalized_text = text.lower()
    if re.search(r"\b(?:multiple|several)\s+(?:visible\s+)?(?:playing\s+)?cards?\b", normalized_text):
        return False, "main model output indicates multiple cards"
    json_count = _count_cards_in_json(result)
    card_details = moondream_provider.card_response_details(result)
    extracted_count = len(card_details.get("deduped_cards") or [])
    count = json_count if json_count is not None else _card_count_from_text(normalized_text)
    if count is not None and count != 1:
        return False, f"main model output indicates {count} cards"
    if json_count == 1:
        return True, "json indicates exactly one visible card"
    if count == 1 and extracted_count <= 1:
        return True, "text says exactly one visible card"
    if extracted_count == 1 and card_details.get("card_result_clean"):
        return True, "model output clearly identifies exactly one card"
    if extracted_count > 1:
        return False, "main model output lists more than one card"
    if any(term in normalized_text for term in _CARD_LIST_TERMS):
        return False, "card count missing from card-list output"
    return False, "no clear single-card identification"


def should_skip_evaluator_for_single_card(
    result: Any, tool_name: Any, task_type: Any
) -> bool:
    return _single_card_evaluator_skip_reason(result, tool_name, task_type)[0]


def _streaming_acceptance_guard(goal: Any, response: str) -> bool:
    text = " ".join(str(response or "").split())
    goal_text = " ".join(str(goal or "").split()).lower()
    if not text or len(text) > 320 or len(text.split()) > 55:
        return False
    if _STREAMING_UNCERTAINTY_PATTERN.search(text):
        return False
    if any(term in goal_text for term in ("card", "rank", "suit", "poker")):
        return bool(_CARD_RANK_PATTERN.search(text) and _CARD_SUIT_PATTERN.search(text))
    stopwords = {
        "what", "which", "where", "please", "briefly", "identify", "describe",
        "show", "shows", "image", "visible", "using", "this", "that", "with",
        "from", "task", "answer", "tell", "user", "current",
    }
    goal_terms = {
        token for token in re.findall(r"[a-z0-9]+", goal_text)
        if len(token) >= 4 and token not in stopwords
    }
    response_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(text.split()) >= 3 and bool(goal_terms & response_terms)


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
        logger.info("[Moondream] cooldown active remaining_seconds=%.1f; continuing cascade", remaining)
        raise RuntimeError("Moondream Cloud cooldown is active")

    model = os.environ.get("MOONDREAM_MODEL", "").strip() or profile.model
    timeout = float(os.environ.get("MOONDREAM_TIMEOUT_SECONDS", "2.0"))
    source_type = moondream_provider.input_type(image_items[0])
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
        "[Moondream] prompt_adapter=extractive_short prompt_source=%s "
        "prompt_chars_original=%d prompt_chars_sent=%d",
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
        if card_mode:
            card_details = moondream_provider.card_response_details(raw_text)
            text = card_details["final_response"]
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
                "[Moondream] final_card_response=%s",
                json.dumps(text, ensure_ascii=False),
            )
            logger.info(
                "[Moondream] card_result_valid=%s card_result_clean=%s",
                str(card_details["card_result_valid"]).lower(),
                str(card_details["card_result_clean"]).lower(),
            )
        else:
            text = " ".join(raw_text.split())
        if not text:
            raise RuntimeError("Moondream returned an empty or malformed response")
        postprocess = (
            "card_rank_suit_only"
            if card_mode
            else "whitespace_only" if text != raw_text.strip() else "none"
        )
        logger.info("[Moondream] postprocess=%s", postprocess)
        _record_moondream_success()
        logger.info(
            "[Moondream] response status=200 request_id=%s latency_ms=%.1f text_chars=%d",
            getattr(completion, "_request_id", None) or "unavailable",
            (time.perf_counter() - started) * 1000,
            len(text),
        )
        artifact = {"text": text}
        if card_mode:
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
        logger.info("[Moondream] continuing cascade")
        raise


def _google_vision_executor(profile, messages, images, metadata) -> ImplementationResult:
    image_items = list(images or [])
    if not image_items:
        return ImplementationResult(_simple_response("No image was provided."), {"text": ""})
    logger.info("[Google Vision Auth] method=application_default_credentials_oauth_rest")
    logger.info(
        "[Google Vision Auth] GOOGLE_APPLICATION_CREDENTIALS configured=%s",
        bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")),
    )
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
    scope = "https://www.googleapis.com/auth/cloud-platform"
    logger.info("[Google Vision Auth] oauth_scope=%s", scope)
    try:
        credentials, project_id = google.auth.default(
            scopes=[scope]
        )
        project_id = project_id or getattr(credentials, "project_id", None)
        logger.info("[Google Vision Auth] default_credentials_loaded=true")
        logger.info(
            "[Google Vision Auth] project_id=%s",
            project_id or "<unknown>",
        )
        logger.info(
            "[Google Vision Auth] principal=%s",
            getattr(credentials, "service_account_email", None) or "<unknown>",
        )
    except Exception:
        logger.exception("[Google Vision Auth] default_credentials_loaded=false")
        raise
    try:
        if not credentials.valid or not credentials.token:
            credentials.refresh(GoogleAuthRequest())
        if not credentials.token:
            raise RuntimeError("Application Default Credentials did not produce an access token")
        logger.info("[Google Vision Auth] oauth_token_created=true")
    except Exception:
        logger.exception("[Google Vision Auth] oauth_token_created=false")
        raise

    # This provider intentionally uses the Vision REST endpoint, not google.cloud.vision.
    logger.info(
        "[Google Vision Auth] vision_client_created=false reason=direct_rest_provider "
        "rest_request_ready=true"
    )
    body = json.dumps({"requests": [{"image": {"content": base64.b64encode(_image_bytes(image_items[0])).decode("ascii")}, "features": [{"type": "TEXT_DETECTION"}]}]}).encode("utf-8")
    url = "https://vision.googleapis.com/v1/images:annotate"
    logger.info("[Google Vision HTTP] endpoint=%s", url)
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    if project_id:
        headers["x-goog-user-project"] = project_id
        logger.info("[Google Vision Auth] quota_project_id=%s", project_id)
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=metadata.get("timeout", 30)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "[Google Vision HTTP] status=%s reason=%s response_body=%s",
            exc.code,
            exc.reason,
            response_body,
        )
        raise RuntimeError(
            f"Google Vision HTTP {exc.code} {exc.reason}: {response_body}"
        ) from exc
    first = (payload.get("responses") or [{}])[0]
    if first.get("error"):
        raise RuntimeError(first["error"].get("message", "Google Vision OCR failed"))
    annotations = first.get("textAnnotations") or []
    text = str(annotations[0].get("description", "")).strip() if annotations else ""
    if not text:
        uncertainty = "The OCR stage could not confidently extract readable text."
        return ImplementationResult(
            _simple_response(uncertainty),
            {"text": "", "accepted": False, "error": "no_text_extracted"},
        )
    return ImplementationResult(_simple_response(text), {"text": text, "accepted": True})


def _mistral_ocr_executor(profile, messages, images, metadata) -> ImplementationResult:
    image_items = list(images or [])
    if not image_items:
        logger.warning("[Mistral OCR] failed reason=no_image")
        raise RuntimeError("no_image")

    from PIL import Image
    image = Image.open(io.BytesIO(_image_bytes(image_items[0]))).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    jpeg_bytes = buffer.getvalue()
    image_data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")
    model = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
    logger.info(
        "[Mistral OCR] model=%s input=image_data_uri payload_kb=%.1f",
        model,
        len(jpeg_bytes) / 1024,
    )

    try:
        from mistralai.client import Mistral
    except Exception as exc:
        logger.warning(
            "[Mistral OCR] import_failed path=mistralai.client.Mistral error=%s",
            exc,
        )
        raise RuntimeError("mistral_ocr_import_failed") from exc

    try:
        client = Mistral()
        ocr_response = client.ocr.process(
            model=model,
            document={
                "type": "image_url",
                "image_url": image_data_uri,
            },
        )
    except Exception as exc:
        logger.warning("[Mistral OCR] sdk_call_failed error=%s", exc)
        raise RuntimeError("mistral_ocr_failed") from exc

    text = "\n".join(
        str(page.markdown).strip()
        for page in (getattr(ocr_response, "pages", None) or [])
        if getattr(page, "markdown", None) and str(page.markdown).strip()
    ).strip()
    if not text:
        logger.warning("[Mistral OCR] failed reason=empty_text")
        raise RuntimeError("empty_text")
    return ImplementationResult(_simple_response(text), {"text": text, "accepted": True})


def _tesseract_executor(profile, messages, images, metadata) -> ImplementationResult:
    image_items = list(images or [])
    if not image_items:
        logger.info("[OCR] tesseract insufficient reason=no_image")
        raise RuntimeError("no_image")
    try:
        import pytesseract
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning("[OCR] tesseract insufficient reason=pytesseract_unavailable")
        raise RuntimeError("pytesseract_unavailable") from exc

    from PIL import Image
    image = Image.open(io.BytesIO(_image_bytes(image_items[0]))).convert("RGB")
    try:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        not_found = getattr(pytesseract, "TesseractNotFoundError", ())
        reason = "tesseract_binary_unavailable" if not_found and isinstance(exc, not_found) else "tesseract_error"
        logger.warning("[OCR] tesseract insufficient reason=%s", reason)
        raise RuntimeError(reason) from exc

    raw_tokens = data.get("text", []) if isinstance(data, dict) else []
    text = " ".join(str(token).strip() for token in raw_tokens if str(token).strip()).strip()
    if not text:
        logger.info("[OCR] tesseract insufficient reason=no_text")
        raise RuntimeError("no_text")

    confidences = []
    for value in data.get("conf", []) if isinstance(data, dict) else []:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            confidences.append(confidence)
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    logger.info("[OCR] tesseract produced text_chars=%s", len(text))
    artifact = {"text": text, "accepted": True}
    if mean_confidence is not None:
        artifact["confidence"] = mean_confidence
    return ImplementationResult(_simple_response(text), artifact)


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
    if not target_labels:
        logger.info("[Target Grounding] detector=default reason=no_target_labels")
        return profile.model_name or YOLO11_MODEL, "default"
    if all(label in COCO_CLASSES for label in target_labels):
        logger.info(
            "[Target Grounding] detector=yolo11 reason=all_targets_in_coco target_labels=%s",
            list(target_labels),
        )
        return YOLO11_MODEL, "yolo11"
    logger.info(
        "[Target Grounding] detector=yoloworld reason=non_coco_targets target_labels=%s",
        list(target_labels),
    )
    return YOLOWORLD_MODEL, "yoloworld"


def _mentions_exit(value: Any) -> bool:
    text = str(value or "").lower()
    return any(label in text for label in EXIT_TARGET_LABELS)


def _label_matches_targets(label: Any, target_labels: Sequence[str]) -> bool:
    normalized = str(label or "").strip().lower()
    return any(normalized in TARGET_LABEL_ALIASES.get(target, {target}) for target in target_labels)


def _filter_target_artifact(artifact: Any, target_labels: Sequence[str]) -> Dict[str, Any]:
    source = artifact if isinstance(artifact, dict) else {}
    detections = source.get("detections") if isinstance(source.get("detections"), list) else []
    matching = [
        item for item in detections
        if isinstance(item, dict) and _label_matches_targets(item.get("label"), target_labels)
    ]
    return {
        **{key: value for key, value in source.items() if key != "detections"},
        "detections": matching,
        "target_labels": list(target_labels),
        "matching_detection": bool(matching),
    }


def _artifact_is_useful(artifact: Any) -> bool:
    """Return whether a prior stage produced usable positive information."""
    if artifact is None:
        return False
    if isinstance(artifact, str):
        return bool(artifact.strip()) and not artifact.strip().lower().startswith(
            ("no ", "not found", "couldn't", "could not", "failed")
        )
    if isinstance(artifact, (list, tuple)):
        return bool(artifact)
    if not isinstance(artifact, dict):
        return bool(artifact)
    if artifact.get("accepted") is False or artifact.get("low_confidence") is True:
        return False
    confidence = artifact.get("confidence")
    if isinstance(confidence, str) and confidence.strip().lower() in {"low", "uncertain"}:
        return False
    if "matching_detection" in artifact and not artifact.get("matching_detection"):
        return False
    for key in ("detections", "objects", "regions", "items"):
        if key in artifact:
            return isinstance(artifact[key], (list, tuple)) and bool(artifact[key])
    for key in ("text", "ocr_text", "content"):
        if key in artifact:
            return _artifact_is_useful(artifact[key])
    ignored = {"target_labels", "confidence", "accepted", "low_confidence", "error"}
    return any(value not in (None, "", [], {}) for key, value in artifact.items() if key not in ignored)


def _previous_stage_text(metadata: Mapping[str, Any], artifact: Any) -> str:
    """Extract text-only prior-stage context for the non-visual evaluator."""
    values: List[str] = []
    legacy = metadata.get("previous_stage_output")
    if isinstance(legacy, str) and legacy.strip():
        values.append(legacy.strip())
    if isinstance(artifact, str) and artifact.strip():
        values.append(artifact.strip())
    elif isinstance(artifact, dict):
        for key in ("text", "ocr_text", "content", "description", "response"):
            value = artifact.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return "\n".join(dict.fromkeys(values))


def _yolo_executor(profile, messages, images, metadata) -> ImplementationResult:
    image_items = list(images or [])
    if not image_items:
        return ImplementationResult(_simple_response("No image was provided."), {"detections": []})
    from PIL import Image
    from ultralytics import YOLO
    image = Image.open(io.BytesIO(_image_bytes(image_items[0]))).convert("RGB")
    cache = metadata.get("model_cache") if isinstance(metadata.get("model_cache"), dict) else {}
    target_labels = _target_labels(metadata)
    model_name, detector = _object_detector_model(profile, target_labels)
    model = cache.get(model_name) or YOLO(model_name)
    cache[model_name] = model
    if detector == "yoloworld":
        model.set_classes(target_labels)
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


def _groundingdino_executor(profile, messages, images, metadata) -> ImplementationResult:
    labels = metadata.get("target_labels") or metadata.get("targets")
    labels = [labels] if isinstance(labels, str) else labels
    if not isinstance(labels, list) or not labels:
        raise RuntimeError("GroundingDINO requires metadata.target_labels")
    image_items = list(images or [])
    if not image_items:
        return ImplementationResult(_simple_response("No image was provided."), {"detections": []})
    from PIL import Image
    from transformers import pipeline
    image = Image.open(io.BytesIO(_image_bytes(image_items[0]))).convert("RGB")
    detector = pipeline("zero-shot-object-detection", model=profile.model_name or "IDEA-Research/grounding-dino-tiny")
    detections = []
    for item in detector(image, candidate_labels=[str(label) for label in labels]):
        box = item.get("box") or {}
        bbox = [box.get("xmin", 0), box.get("ymin", 0), box.get("xmax", 0), box.get("ymax", 0)]
        detections.append({"label": item.get("label", "object"), "bbox": bbox, "location": _location(bbox, image.width, image.height)})
    text = "; ".join(f"{item['label']} at {item['location']}" for item in detections) or "No requested object was detected."
    return ImplementationResult(_simple_response(text), {"detections": detections})


IMPLEMENTATION_EXECUTORS: Dict[str, ImplementationExecutor] = {
    "model": _model_executor,
    "moondream_cloud": _moondream_cloud_executor,
    "mistral_ocr": _mistral_ocr_executor,
    "tesseract": _tesseract_executor,
    "google_vision": _google_vision_executor,
    "yolo": _yolo_executor,
    "groundingdino": _groundingdino_executor,
}


def system_llm_call(messages=None, images=None, metadata=None):
    """Call the YAML-configured infrastructure model without capability routing."""
    config = load_global_execution_config()
    implementations = load_implementation_profiles()
    _log_global_execution_config(config, implementations)
    implementation = config["system_model"]
    profile = implementations[implementation]
    if profile.kind != "model" or not profile.model:
        raise ExecutionPolicyError(
            f"System implementation {implementation!r} must define kind=model and model"
        )
    logger.info(
        "[Execution Policy] capability=system selected implementation=%s",
        implementation,
    )
    return call_model(profile.model, messages or [], images=images, metadata=metadata)


def single_stage_llm_call(task=None, messages=None, images=None, metadata=None):
    """Bypass planning and routing and call the configured default model once."""
    config = load_global_execution_config()
    implementations = load_implementation_profiles()
    _log_global_execution_config(config, implementations)
    implementation = config["default_llm_when_routing_disabled"]
    profile = implementations[implementation]
    if profile.kind != "model" or not profile.model:
        raise ExecutionPolicyError(
            f"Default implementation {implementation!r} must define kind=model and model"
        )
    call_messages = list(messages or [])
    call_messages.insert(0, {"role": "system", "content": AUDIO_RESPONSE_PROMPT})
    streaming = bool((metadata or {}).get("streaming") or STREAMING_EXECUTION_CONTEXT.get())
    if streaming:
        call_messages.insert(0, {"role": "system", "content": STREAMING_RESPONSE_PROMPT})
        logger.info("[Streaming] concise response prompt enabled")
    if task:
        call_messages.append({"role": "user", "content": str(task)})
    logger.info("[Execution Policy] planner disabled -> single-stage execution")
    logger.info(
        "[Execution Policy] single-stage implementation=%s model=%s",
        implementation,
        profile.model,
    )
    call_metadata = dict(metadata or {})
    call_images = list(images or [])
    total_started = time.perf_counter()
    preprocess_ms = 0.0
    _raise_if_streaming_cancelled(call_metadata, "single_stage", streaming)
    if streaming and call_images:
        call_images, preprocess_ms = _preprocess_streaming_images(
            call_images, DEFAULT_FALLBACK_CAPABILITY,
            call_metadata.get("streaming_original_payload_bytes"),
            call_metadata.get("streaming_original_image"),
        )
    model_started = time.perf_counter()
    response = call_model(
        profile.model, call_messages, images=call_images, metadata=call_metadata
    )
    model_ms = (time.perf_counter() - model_started) * 1000
    _raise_if_streaming_cancelled(call_metadata, "single_stage_result", streaming)
    text = _single_line_audio_text(_response_text(response))
    if streaming:
        normalized_model = profile.model.lower()
        gemini_ms = model_ms if "gemini" in normalized_model else 0.0
        gpt4o_ms = model_ms if "gpt-4o" in normalized_model or "gpt4o" in normalized_model else 0.0
        logger.info(
            "[Streaming Timing] execution=%s total_ms=%.1f image_preprocess_ms=%.1f "
            "moondream_ms=0.0 evaluator1_ms=0.0 gemini_ms=%.1f evaluator2_ms=0.0 "
            "gpt4o_ms=%.1f selected=%s",
            call_metadata.get("streaming_execution_id", "unknown"),
            (time.perf_counter() - total_started) * 1000,
            preprocess_ms, gemini_ms, gpt4o_ms, implementation,
        )
    return {
        "response": text,
        "artifact": {"text": text},
        "implementation": implementation,
        "capability": DEFAULT_FALLBACK_CAPABILITY,
    }


def copilot_llm_call(
    capability=None,
    messages=None,
    images=None,
    metadata=None,
    task=None,
    task_category=None,
    goal=None,
):
    """Execute exactly one capability using its first configured implementation."""
    requested = capability or task_category or DEFAULT_FALLBACK_CAPABILITY
    declared = normalize_capability_name(requested)
    if declared != str(requested).strip().lower():
        logger.warning("[Execution Policy] normalized legacy capability %r to %r", requested, declared)
    implementations = load_implementation_profiles()
    global_config = load_global_execution_config()
    _log_global_execution_config(global_config, implementations)
    if not global_config["planner_enabled"]:
        return single_stage_llm_call(
            task=goal or task,
            messages=messages,
            images=images,
            metadata=metadata,
        )
    policies = load_execution_policies()
    if declared not in policies:
        raise ExecutionPolicyError(
            f"Unknown capability {declared!r}; supported capabilities are: {sorted(policies)}"
        )
    policy = policies[declared]
    logger.info("[Execution Policy] planner enabled -> executing staged pipeline")
    if global_config["routing_enabled"]:
        logger.info("[Execution Policy] routing enabled -> selecting implementations")
        candidates = policy["candidates"]
        evaluator_name = policy.get("evaluator")
    else:
        logger.info("[Execution Policy] routing disabled -> using one fixed implementation per stage")
        candidates = [global_config["default_llm_when_routing_disabled"]]
        evaluator_name = None

    call_metadata = dict(metadata or {})
    call_metadata.setdefault("model_cache", _IMPLEMENTATION_MODEL_CACHE)
    call_metadata["capability"] = declared
    step_index = call_metadata.get("step_index", "unknown")
    prior_results_supplied = bool(
        call_metadata.get("previous_stage_artifact")
        or call_metadata.get("previous_stage_artifacts")
        or call_metadata.get("previous_stage_output")
    )
    streaming_log = bool(
        call_metadata.get("streaming") or STREAMING_EXECUTION_CONTEXT.get()
    )
    if task and "task_text" not in call_metadata:
        call_metadata["task_text"] = task
    call_messages = list(messages or [])
    call_messages.insert(0, {"role": "system", "content": AUDIO_RESPONSE_PROMPT})
    target_labels = _target_labels(call_metadata)
    grounding_context = [goal, task, call_metadata.get("goal"), call_metadata.get("task_text")]
    grounding_context.extend(message.get("content") for message in call_messages if isinstance(message, dict))
    card_task_mode, card_task_reason = moondream_provider.card_mode_reason(
        " ".join(str(value) for value in grounding_context if value)
    )
    if card_task_mode:
        logger.info("[Card Normalizer] card_mode=true reason=%s", card_task_reason)
    if not target_labels and any(_mentions_exit(value) for value in grounding_context):
        target_labels = list(EXIT_TARGET_LABELS)
        call_metadata["target_labels"] = target_labels
    if declared == "object_detection_localization":
        logger.info("[Target Grounding] object_detection_localization target_labels=%s", target_labels)
    previous_artifact = call_metadata.get("previous_stage_artifact")
    if declared == "navigation":
        if not target_labels and isinstance(previous_artifact, dict):
            target_labels = _target_labels(previous_artifact)
            call_metadata["target_labels"] = target_labels
        if target_labels and previous_artifact is not None:
            previous_artifact = _filter_target_artifact(previous_artifact, target_labels)
            call_metadata["previous_stage_artifact"] = previous_artifact
            logger.info("[Target Grounding] navigation target_artifact=%s", previous_artifact)
        legacy_output = call_metadata.get("previous_stage_output")
        if target_labels and previous_artifact is None and legacy_output is not None:
            logger.info("[Target Grounding] navigation target_artifact=%s", legacy_output)
            matching_alias = any(
                alias in str(legacy_output).lower()
                for target in target_labels
                for alias in TARGET_LABEL_ALIASES.get(target, {target})
            )
            if not matching_alias:
                previous_artifact = None
        target_text = ", ".join(target_labels) or "the requested destination"
        call_messages.insert(0, {
            "role": "system",
            "content": NAVIGATION_SYSTEM_PROMPT +
            f" The navigation target is: {target_text}. "
            "Never substitute another detected object for this target.",
        })
    if _artifact_is_useful(previous_artifact):
        call_messages.append({
            "role": "user",
            "content": "Useful previous-stage artifact (additional context): "
            + json.dumps(previous_artifact, ensure_ascii=False, default=str),
        })
    elif previous_artifact is not None or call_metadata.get("previous_stage_output") is not None:
        logger.info(
            "[Stage Handoff] previous artifact unusable -> dropping artifact and using original image"
        )
    if goal:
        call_metadata["goal"] = str(goal)
        call_messages.append({"role": "user", "content": str(goal)})
    call_images = list(images or TOOL_EXECUTION_IMAGES.get() or [])
    cascade_started = time.perf_counter()
    timing = {
        "image_preprocess_ms": 0.0,
        "moondream_ms": 0.0,
        "evaluator1_ms": 0.0,
        "gemini_ms": 0.0,
        "evaluator2_ms": 0.0,
        "gpt4o_ms": 0.0,
    }
    if streaming_log and call_images:
        candidate_kinds = {
            implementations[name].kind for name in candidates if name in implementations
        }
        if declared == "ocr" or candidate_kinds & {"model", "moondream_cloud"}:
            call_images, timing["image_preprocess_ms"] = _preprocess_streaming_images(
                call_images, declared,
                call_metadata.get("streaming_original_payload_bytes"),
                call_metadata.get("streaming_original_image"),
            )

    execution_label = call_metadata.get("streaming_execution_id", "unknown")

    def record_candidate_timing(name: str, elapsed_ms: float) -> None:
        normalized = name.lower()
        if "moondream" in normalized:
            timing["moondream_ms"] += elapsed_ms
        elif "gemini" in normalized:
            timing["gemini_ms"] += elapsed_ms
        elif "gpt4o" in normalized or "gpt-4o" in normalized:
            timing["gpt4o_ms"] += elapsed_ms

    def log_streaming_timing(selected_name: str) -> None:
        if not streaming_log:
            return
        logger.info(
            "[Streaming Timing] execution=%s total_ms=%.1f image_preprocess_ms=%.1f "
            "moondream_ms=%.1f evaluator1_ms=%.1f gemini_ms=%.1f "
            "evaluator2_ms=%.1f gpt4o_ms=%.1f selected=%s",
            execution_label, (time.perf_counter() - cascade_started) * 1000,
            timing["image_preprocess_ms"], timing["moondream_ms"],
            timing["evaluator1_ms"], timing["gemini_ms"],
            timing["evaluator2_ms"], timing["gpt4o_ms"], selected_name,
        )

    output = None
    implementation = ""
    best_output = None
    best_implementation = ""
    selected = False
    evaluator_count = 0
    for index, candidate in enumerate(candidates):
        if streaming_log and _streaming_cancelled(call_metadata):
            log_streaming_timing("cancelled")
        _raise_if_streaming_cancelled(call_metadata, f"candidate:{candidate}", streaming_log)
        if streaming_log:
            logger.info(
                "[Streaming] implementation candidate=%s capability=%s",
                candidate,
                declared,
            )
        logger.info("[Execution Policy] capability=%s trying=%s", declared, candidate)
        logger.info("[Execution Policy] capability=%s candidate=%s", declared, candidate)
        started_at = time.perf_counter()
        if declared == "ocr":
            logger.info("[OCR] trying=%s", candidate)
        try:
            profile = implementations[candidate]
            executor = IMPLEMENTATION_EXECUTORS.get(profile.kind)
            if executor is None:
                raise ExecutionPolicyError(
                    f"No executor for implementation kind {profile.kind!r}"
                )
            candidate_output = executor(profile, call_messages, call_images, call_metadata)
        except Exception as exc:
            record_candidate_timing(candidate, (time.perf_counter() - started_at) * 1000)
            if isinstance(exc, StreamingExecutionCancelled):
                log_streaming_timing("cancelled")
                raise
            if declared == "ocr":
                logger.info(
                    "[OCR] implementation=%s latency_ms=%.1f",
                    candidate,
                    (time.perf_counter() - started_at) * 1000,
                )
            logger.warning(
                "[Execution Policy] capability=%s implementation=%s failed=%s",
                declared,
                candidate,
                exc,
            )
            continue
        record_candidate_timing(candidate, (time.perf_counter() - started_at) * 1000)
        if not global_config["routing_enabled"]:
            logger.info(
                "[Separate Steps] step_index=%s capability=%s selected_fixed_implementation=%s "
                "latency_ms=%.1f prior_results_supplied=%s model_calls=1 evaluator_calls=0 "
                "cascade_escalations=0",
                step_index,
                declared,
                candidate,
                (time.perf_counter() - started_at) * 1000,
                str(prior_results_supplied).lower(),
            )
        if streaming_log and _streaming_cancelled(call_metadata):
            log_streaming_timing("cancelled")
        _raise_if_streaming_cancelled(
            call_metadata, f"evaluator_after:{candidate}", streaming_log
        )
        response_text = _single_line_audio_text(_response_text(candidate_output.response))
        raw_main_model_output = response_text
        tool_name = call_metadata.get("tool_name", "")
        task_type = call_metadata.get("task_type") or declared
        card_result_valid = False
        card_result_clean = False
        if card_task_mode:
            upstream_artifact = (
                dict(candidate_output.artifact)
                if isinstance(candidate_output.artifact, dict)
                else {}
            )
            raw_card_response = upstream_artifact.get("raw_response", response_text)
            raw_main_model_output = raw_card_response
            logger.info(
                "[Card Normalizer] implementation=%s raw_response=%s",
                candidate,
                json.dumps(raw_card_response, ensure_ascii=False),
            )
            if "card_result_clean" in upstream_artifact:
                card_details = {
                    key: upstream_artifact[key]
                    for key in (
                        "raw_response", "extracted_cards", "deduped_cards",
                        "discarded_card_tokens", "conflicting_suits",
                        "compact_notation_detected", "mirrored_corner_filter_applied",
                        "card_result_valid", "card_result_clean", "final_response",
                    )
                    if key in upstream_artifact
                }
            else:
                card_details = moondream_provider.card_response_details(raw_card_response)
            response_text = card_details["final_response"]
            card_result_valid = card_details["card_result_valid"]
            card_result_clean = card_details["card_result_clean"]
            logger.info(
                "[Card Normalizer] compact_notation_detected=%s extracted_cards=%s "
                "discarded_card_tokens=%s conflicting_suits=%s final_card_response=%s "
                "card_result_valid=%s card_result_clean=%s",
                str(card_details["compact_notation_detected"]).lower(),
                json.dumps(card_details["extracted_cards"], ensure_ascii=False),
                json.dumps(card_details["discarded_card_tokens"], ensure_ascii=False),
                json.dumps(card_details["conflicting_suits"], ensure_ascii=False),
                json.dumps(response_text, ensure_ascii=False),
                str(card_result_valid).lower(),
                str(card_result_clean).lower(),
            )
            artifact = upstream_artifact
            artifact.update(card_details)
            candidate_output = ImplementationResult(_simple_response(response_text), artifact)
        if declared == "ocr":
            logger.info(
                "[OCR] implementation=%s latency_ms=%.1f text_chars=%s",
                candidate,
                (time.perf_counter() - started_at) * 1000,
                len(response_text),
            )
        if not response_text:
            logger.warning(
                "[Execution Policy] capability=%s implementation=%s failed=empty response",
                declared,
                candidate,
            )
            continue

        best_output = candidate_output
        best_implementation = candidate
        logger.info(
            "[Execution Policy] capability=%s implementation=%s response:\n%s",
            declared,
            candidate,
            response_text,
        )

        evaluator_available = bool(
            evaluator_name and len(candidates) > 1 and index < len(candidates) - 1
        )
        skip_evaluator_for_single_card, single_card_skip_reason = (
            _single_card_evaluator_skip_reason(raw_main_model_output, tool_name, task_type)
        )
        evaluator_skipped = bool(evaluator_available and skip_evaluator_for_single_card)
        if not evaluator_available:
            single_card_skip_reason = "evaluator unavailable for this candidate"
        logger.info(
            "[Evaluator] tool_name=%s task_type=%s raw_main_model_output=%s "
            "skipped=%s reason=%s",
            json.dumps(str(tool_name), ensure_ascii=False),
            json.dumps(str(task_type), ensure_ascii=False),
            json.dumps(raw_main_model_output, ensure_ascii=False),
            str(evaluator_skipped).lower(),
            single_card_skip_reason,
        )
        if evaluator_skipped:
            logger.info(
                "[Evaluator] skipped: single visible card for visual_representation_understanding"
            )
        if evaluator_available and not evaluator_skipped:
            evaluator_count += 1
            evaluator_started = time.perf_counter()
            task_goal = goal or call_metadata.get("goal") or call_metadata.get("task_text") or ""
            debug_reason = os.environ.get("EVALUATOR_DEBUG_REASON", "false").lower() in {
                "1", "true", "yes", "on",
            }
            logger.info("[Evaluator] task_goal=%s", json.dumps(str(task_goal), ensure_ascii=False))
            logger.info(
                "[Evaluator] candidate_response=%s",
                json.dumps(response_text, ensure_ascii=False),
            )
            _raise_if_streaming_cancelled(
                call_metadata, f"evaluator:{evaluator_name}", streaming_log
            )
            try:
                evaluator_profile = implementations[evaluator_name]
                evaluator = IMPLEMENTATION_EXECUTORS.get(evaluator_profile.kind)
                if evaluator is None:
                    raise ExecutionPolicyError(
                        f"No executor for evaluator implementation kind {evaluator_profile.kind!r}"
                    )
                previous_text = _previous_stage_text(call_metadata, previous_artifact)
                evaluator_metadata = {
                    "temperature": 0,
                    "max_tokens": 80 if debug_reason else 3,
                    "capability": declared,
                    "evaluator": True,
                }
                logger.info("[Evaluator] image_included=false")
                evaluation_prompt = EVALUATION_PROMPT.format(
                    capability=declared,
                    goal=task_goal,
                    previous_text=previous_text,
                    response=response_text,
                )
                if debug_reason:
                    evaluation_prompt += (
                        "\nDebug output format:\nDECISION: YES or NO\n"
                        "REASON: one short sentence"
                    )
                evaluation = evaluator(
                    evaluator_profile,
                    [{
                        "role": "user",
                        "content": evaluation_prompt,
                    }],
                    [],
                    evaluator_metadata,
                )
                raw_evaluation = _response_text(evaluation.response).strip()
                if debug_reason:
                    decision_match = re.search(
                        r"DECISION:\s*(YES|NO)\b", raw_evaluation, re.IGNORECASE
                    )
                    reason_match = re.search(
                        r"REASON:\s*(.+)", raw_evaluation, re.IGNORECASE | re.DOTALL
                    )
                    raw_decision = decision_match.group(1).upper() if decision_match else ""
                    debug_reason_text = reason_match.group(1).strip() if reason_match else "unavailable"
                else:
                    raw_decision = raw_evaluation.upper()
                    debug_reason_text = ""
                if raw_decision not in {"YES", "NO"}:
                    raise ValueError("evaluator did not return YES or NO")
                decision = raw_decision
            except Exception as exc:
                if evaluator_count <= 2:
                    timing[f"evaluator{evaluator_count}_ms"] += (
                        time.perf_counter() - evaluator_started
                    ) * 1000
                if isinstance(exc, StreamingExecutionCancelled):
                    log_streaming_timing("cancelled")
                    raise
                logger.warning(
                    "[Execution Policy] capability=%s evaluator=%s decision=FAILED error=%s",
                    declared,
                    evaluator_name,
                    exc,
                )
                continue

            if evaluator_count <= 2:
                timing[f"evaluator{evaluator_count}_ms"] += (
                    time.perf_counter() - evaluator_started
                ) * 1000
            if streaming_log and _streaming_cancelled(call_metadata):
                log_streaming_timing("cancelled")
            _raise_if_streaming_cancelled(
                call_metadata, f"evaluator_result:{evaluator_name}", streaming_log
            )

            logger.info(
                "[Execution Policy] capability=%s evaluator=%s decision=%s",
                declared,
                evaluator_name,
                decision,
            )
            logger.info("[Evaluator] decision=%s", decision)
            if debug_reason:
                logger.info(
                    "[Evaluator] reason=%s",
                    json.dumps(debug_reason_text, ensure_ascii=False),
                )
            elif logger.isEnabledFor(logging.DEBUG):
                reason_started = time.perf_counter()
                try:
                    _raise_if_streaming_cancelled(
                        call_metadata, f"evaluator_reason:{evaluator_name}", streaming_log
                    )
                    reason_output = evaluator(
                        evaluator_profile,
                        [{
                            "role": "user",
                            "content": EVALUATION_REASON_PROMPT.format(
                                capability=declared,
                                goal=task_goal,
                                previous_text=previous_text,
                                response=response_text,
                                decision=decision,
                            ),
                        }],
                        [],
                        {
                            "temperature": 0,
                            "max_tokens": 60,
                            "capability": declared,
                            "evaluator_reason": True,
                        },
                    )
                    reason = " ".join(_response_text(reason_output.response).split())
                    logger.debug("[Evaluator] reason=%s", json.dumps(reason, ensure_ascii=False))
                    logger.debug(
                        "[Evaluator] decision=%s reason=%s",
                        decision, json.dumps(reason, ensure_ascii=False),
                    )
                except Exception as exc:
                    if isinstance(exc, StreamingExecutionCancelled):
                        log_streaming_timing("cancelled")
                        raise
                    logger.debug(
                        "[Evaluator] reason=%s",
                        json.dumps(f"reason unavailable: {exc}"),
                    )
                finally:
                    if evaluator_count <= 2:
                        timing[f"evaluator{evaluator_count}_ms"] += (
                            time.perf_counter() - reason_started
                        ) * 1000
            if streaming_log:
                logger.info(
                    "[Streaming] evaluator=%s capability=%s decision=%s",
                    evaluator_name,
                    declared,
                    decision,
                )
            if decision == "NO" and streaming_log and _streaming_acceptance_guard(
                task_goal, response_text
            ):
                decision = "YES"
                logger.info(
                    "[Streaming] acceptance_guard=accepted capability=%s candidate=%s",
                    declared, candidate,
                )
                logger.info("[Evaluator] decision=YES source=streaming_acceptance_guard")
            if decision == "NO":
                continue

        output = candidate_output
        implementation = candidate
        selected = True
        logger.info(
            "[Execution Policy] capability=%s selected implementation=%s",
            declared,
            candidate,
        )
        break

    if not selected:
        if best_output is not None:
            output = best_output
            implementation = best_implementation
            logger.info(
                "[Execution Policy] capability=%s fallback=%s",
                declared,
                best_implementation,
            )
        else:
            fallback_text = "The previous stage could not produce a reliable result."
            output = ImplementationResult(
                _simple_response(fallback_text),
                {"text": fallback_text, "accepted": False, "error": "stage_failed"},
            )
            implementation = "fallback"
            logger.warning("[Execution Policy] capability=%s fallback=none", declared)
    final_text = _single_line_audio_text(_response_text(output.response))
    if card_task_mode:
        final_details = moondream_provider.card_response_details(final_text)
        if final_details["card_result_valid"]:
            final_text = final_details["final_response"]
            artifact = dict(output.artifact) if isinstance(output.artifact, dict) else {}
            artifact.update(final_details)
            output = ImplementationResult(_simple_response(final_text), artifact)
    artifact = output.artifact
    if artifact is None:
        artifact = {"text": final_text}
    log_streaming_timing(implementation)
    return {
        "response": final_text,
        "artifact": artifact,
        "implementation": implementation,
        "capability": declared,
    }


__all__ = [
    "BACKEND_DIR", "CAPABILITY_PROFILES_PATH", "EXECUTION_POLICY_PATH",
    "LEGACY_CAPABILITY_ALIASES", "NAVIGATION_SYSTEM_PROMPT", "AUDIO_RESPONSE_PROMPT", "TOOL_EXECUTION_IMAGES",
    "EVALUATION_PROMPT", "EVALUATION_REASON_PROMPT", "STREAMING_RESPONSE_PROMPT",
    "ImplementationProfile", "ImplementationResult", "ExecutionPolicyError", "StreamingExecutionCancelled",
    "IMPLEMENTATION_EXECUTORS",
    "normalize_capability_name", "load_capability_descriptions", "load_capability_profiles",
    "load_execution_policies", "load_implementation_profiles", "load_global_execution_config",
    "validate_execution_configuration",
    "system_llm_call", "single_stage_llm_call", "copilot_llm_call",
    "should_skip_evaluator_for_single_card",
]
