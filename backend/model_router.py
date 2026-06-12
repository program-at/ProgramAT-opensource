"""Central routing for ProgramAT model-backed interactions."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False


logger = logging.getLogger(__name__)

MODEL_PROFILES_PATH = Path(__file__).with_name("model_profiles.yaml")
CAPABILITY_PROFILES_PATH = Path(__file__).with_name("capability_profiles.yaml")

DEFAULT_MODEL_PROFILES: Dict[str, Any] = {
    "models": {
        "gemini_flash": {
            "model": "gemini/gemini-3-flash-preview",
            "description": "Fast balanced general model for lightweight multimodal work and conversational follow-up.",
            "routes": [
                "quick conversational follow-up without heavy reasoning",
                "general text and image assistance",
                "lightweight multimodal analysis when speed matters",
                "fallback for short assistant responses",
            ],
            "vision": 4,
            "coding": 3,
            "reasoning": 2,
            "latency": 5,
            "cost": 5,
        },
        "gemini_flash_lite": {
            "model": "gemini/gemini-2.5-flash-lite",
            "description": "Lowest-latency Gemini model for simple text parsing, JSON extraction, issue selection, and concise summaries.",
            "routes": [
                "parse voice transcripts into structured JSON",
                "extract fields from short user requests",
                "choose or retrieve an issue or tool from a list",
                "read text from signs, labels, documents, and images using OCR",
                "extract OCR text from a camera frame for text to speech",
                "summarize logs concisely for text to speech",
                "classify simple commands with minimal latency",
            ],
            "vision": 2,
            "coding": 2,
            "reasoning": 2,
            "latency": 5,
            "cost": 5,
        },
        "llama": {
            "model": os.environ.get("LLAMA_MODEL", "groq/llama-3.1-8b-instant"),
            "description": "Fast Groq-hosted lightweight Llama model for generating, modifying, debugging, and explaining code.",
            "routes": [
                "generate code for a new assistive technology tool",
                "repair bugs or failing tests in Python or TypeScript",
                "refactor backend or mobile app code",
                "reason through implementation tradeoffs",
                "write detailed code explanations",
            ],
            "vision": 2,
            "coding": 4,
            "reasoning": 4,
            "latency": 5,
            "cost": 5,
        },
        "llava": {
            "model": os.environ.get("LLAVA_MODEL", "groq/meta-llama/llama-4-scout-17b-16e-instruct"),
            "description": "Groq-hosted lightweight vision-language model for image understanding through the Groq API.",
            "routes": [
                "answer simple questions about an image using a Groq vision language model",
                "describe visible objects, layout, and text-adjacent visual context from a camera frame",
                "low-latency visual understanding through Groq",
            ],
            "vision": 4,
            "coding": 2,
            "reasoning": 3,
            "latency": 5,
            "cost": 5,
        },
        "gpt4o": {
            "model": "openai/gpt-4o",
            "description": "Vision-first multimodal model for image understanding and visual question answering.",
            "routes": [
                "answer questions about an image",
                "describe a camera frame or scene",
                "identify objects, clothing, text, layout, or spatial relationships in an image",
                "analyze visual accessibility context",
                "combine image input with a user follow-up question",
            ],
            "vision": 5,
            "coding": 4,
            "reasoning": 4,
            "latency": 3,
            "cost": 1,
        },
        "yolo11_detector": {
            "model": "local/yolo11n.pt",
            "backend": "local_detector",
            "description": "Local YOLO11 detector for pure object detection, localization, and counting over COCO-style object classes.",
            "routes": [
                "detect common objects with bounding boxes",
                "localize known object classes in a camera frame",
                "count visible objects using a specialized detector",
                "fast object detection for navigation cues",
            ],
            "vision": 5,
            "coding": 0,
            "reasoning": 0,
            "latency": 5,
            "cost": 5,
        },
        "yolo_world_detector": {
            "model": "local/yolov8s-world.pt",
            "backend": "local_detector",
            "description": "Local open-vocabulary YOLOWorld detector for pure detection/localization of labels outside COCO.",
            "routes": [
                "detect a named object class that is not in COCO",
                "open vocabulary object localization",
                "localize unusual objects with bounding boxes",
            ],
            "vision": 5,
            "coding": 0,
            "reasoning": 1,
            "latency": 4,
            "cost": 5,
        },
        "google_vision_ocr": {
            "model": "google_vision/text_detection",
            "backend": "ocr",
            "description": "Google Cloud Vision OCR for extracting text from signs, labels, documents, menus, and other camera frames.",
            "routes": [
                "extract text from an image",
                "read signs labels menus documents and license plates",
                "OCR text detection with bounding boxes",
            ],
            "vision": 4,
            "coding": 0,
            "reasoning": 0,
            "latency": 4,
            "cost": 3,
        },
    }
}

DEFAULT_CAPABILITY_PROFILES: Dict[str, Any] = {
    "capabilities": {
        "text_parse": {"vision": 0, "coding": 0, "reasoning": 1, "latency": 5},
        "simple_parsing": {"vision": 0, "coding": 0, "reasoning": 1, "latency": 5},
        "tool_retrieval": {"vision": 0, "coding": 0, "reasoning": 2, "latency": 5},
        "image_analysis": {"vision": 5, "coding": 0, "reasoning": 2, "latency": 4},
        "visual_understanding": {"vision": 5, "coding": 0, "reasoning": 2, "latency": 4},
        "visual_reasoning": {"vision": 5, "coding": 0, "reasoning": 4, "latency": 3},
        "ocr": {"vision": 4, "coding": 0, "reasoning": 1, "latency": 4},
        "object_detection": {"vision": 5, "coding": 0, "reasoning": 0, "latency": 5},
        "object_localization": {"vision": 5, "coding": 0, "reasoning": 1, "latency": 5},
        "summarization": {"vision": 0, "coding": 0, "reasoning": 2, "latency": 5},
        "code_generation": {"vision": 0, "coding": 5, "reasoning": 4, "latency": 2},
        "code_repair": {"vision": 0, "coding": 5, "reasoning": 5, "latency": 2},
        "general_reasoning": {"vision": 1, "coding": 1, "reasoning": 4, "latency": 3},
    }
}

CAPABILITY_TO_PROFILE = {
    "text_parse": "gemini_flash_lite",
    "simple_parsing": "gemini_flash_lite",
    "tool_retrieval": "gemini_flash_lite",
    "image_analysis": "gpt4o",
    "visual_understanding": "gpt4o",
    "visual_reasoning": "gpt4o",
    "ocr": "google_vision_ocr",
    "object_detection": "yolo11_detector",
    "object_localization": "yolo11_detector",
    "code_generation": "llama",
    "code_repair": "llama",
    "summarization": "gemini_flash_lite",
    "general_reasoning": "gemini_flash",
}

CAPABILITY_ROUTE_HINTS = {
    "text_parse": "parse transcript structured json extract fields",
    "simple_parsing": "parse short input classify simple commands extract fields",
    "tool_retrieval": "retrieve choose select issue tool from list",
    "image_analysis": "image visual camera frame scene question",
    "visual_understanding": "identify describe image visual camera frame scene objects layout",
    "visual_reasoning": "reason about image visual scene spatial relationships accessibility context",
    "ocr": "read text ocr extract words signs labels documents from image",
    "object_detection": "pure object detection localization counting known objects bounding boxes detector",
    "object_localization": "localize object position bounding box detector navigation target",
    "code_generation": "generate code implement feature",
    "code_repair": "repair bug fix tests debug code",
    "summarization": "summarize summarization concise log summary",
    "general_reasoning": "general reasoning assistant response plan explain decide",
}

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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
}

LLM_FALLBACK_PROFILES = {
    "ocr": "gemini_flash_lite",
    "object_detection": "gpt4o",
    "object_localization": "gpt4o",
}

TASK_CATEGORIES = (
    "simple_parsing",
    "visual_understanding",
    "visual_reasoning",
    "ocr",
    "object_detection",
    "object_localization",
    "summarization",
    "code_generation",
    "general_reasoning",
)

TASK_CATEGORY_ALIASES = {
    "image_analysis": "visual_understanding",
    "ocr": "ocr",
}

_MODEL_PROFILES: Dict[str, Any] | None = None
_CAPABILITY_PROFILES: Dict[str, Any] | None = None
ROUTING_MODE = os.environ.get("ROUTING_MODE", "semantic").strip().lower()
ROUTING_MIN_SCORE = float(os.environ.get("ROUTING_MIN_SCORE", "0.0") or 0.0)


def _routing_mode() -> str:
    return ROUTING_MODE if ROUTING_MODE in {"semantic", "fixed", "score"} else "semantic"


def _normalize_capability(capability: str) -> tuple[str, str]:
    raw_capability = (capability or "").strip()
    lowered = raw_capability.lower()
    return raw_capability, TASK_CATEGORY_ALIASES.get(lowered, lowered)


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return fallback

    try:
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded:
                return loaded
        except ImportError:
            pass

        parsed: Dict[str, Any] = {}
        section_data: Optional[Dict[str, Any]] = None
        item_data: Optional[Dict[str, Any]] = None
        current_list: Optional[List[Any]] = None

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            if indent == 0 and stripped.endswith(":"):
                section_data = parsed.setdefault(stripped[:-1].strip(), {})
                item_data = None
                current_list = None
                continue

            if indent == 2 and stripped.endswith(":") and section_data is not None:
                item_data = section_data.setdefault(stripped[:-1].strip(), {})
                current_list = None
                continue

            if indent >= 4 and stripped.endswith(":") and item_data is not None:
                key = stripped[:-1].strip()
                current_list = item_data.setdefault(key, [])
                continue

            if indent >= 6 and stripped.startswith("- ") and current_list is not None:
                current_list.append(_parse_scalar(stripped[2:]))
                continue

            if indent >= 4 and ":" in stripped and item_data is not None:
                key, raw_value = stripped.split(":", 1)
                item_data[key.strip()] = _parse_scalar(raw_value)
                current_list = None

        return parsed or fallback
    except Exception as exc:
        logger.warning(f"Failed to parse {path}: {exc}")
        return fallback


def _model_profiles() -> Dict[str, Any]:
    global _MODEL_PROFILES
    if _MODEL_PROFILES is None:
        _MODEL_PROFILES = _load_simple_yaml(MODEL_PROFILES_PATH, DEFAULT_MODEL_PROFILES)
        models = _MODEL_PROFILES.get("models", {})
        if isinstance(models, dict):
            if os.environ.get("LLAMA_MODEL") and isinstance(models.get("llama"), dict):
                models["llama"]["model"] = os.environ["LLAMA_MODEL"]
            if os.environ.get("LLAVA_MODEL") and isinstance(models.get("llava"), dict):
                models["llava"]["model"] = os.environ["LLAVA_MODEL"]
    return _MODEL_PROFILES


def _capability_profiles() -> Dict[str, Any]:
    global _CAPABILITY_PROFILES
    if _CAPABILITY_PROFILES is None:
        _CAPABILITY_PROFILES = _load_simple_yaml(CAPABILITY_PROFILES_PATH, DEFAULT_CAPABILITY_PROFILES)
    return _CAPABILITY_PROFILES


def _provider_for_model(model_name: str) -> str:
    raw = (model_name or "").strip()
    if raw.startswith("local/"):
        return "local"
    if raw.startswith("google_vision/"):
        return "google_vision"
    if "/" in raw:
        return raw.split("/", 1)[0]
    if raw.startswith("llama") or raw.startswith("llava"):
        return "ollama"
    if raw.startswith("gemini"):
        return "gemini"
    if raw.startswith("claude"):
        return "anthropic"
    if raw.startswith("gpt"):
        return "openai"
    return "unknown"


def _profile_backend(profile_data: Dict[str, Any]) -> str:
    backend = profile_data.get("backend")
    return str(backend).strip() if backend else "litellm"


def _route_log_context(route_info: Dict[str, Any], tool_name: str | None = None, note: str | None = None) -> str:
    return (
        f"tool_name={tool_name or 'unknown'} task_category={route_info.get('task_category')} "
        f"selected_profile={route_info.get('selected_profile')} selected_backend={_profile_backend(route_info.get('profile_data') or {})} "
        f"selected_model={route_info.get('selected_model')} provider={route_info.get('provider')} "
        f"fallback_model={route_info.get('fallback_model') or 'none'} "
        f"fallback_reason={route_info.get('fallback_reason') or 'none'} "
        f"routing_note={note or 'none'}"
    )


def _resolve_explicit_profile(metadata: Dict[str, Any] | None) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None

    candidate = metadata.get("requested_profile") or metadata.get("profile")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    requested_model = metadata.get("requested_model") or metadata.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        return None

    requested_model = requested_model.strip()
    for profile_name, profile_data in _model_profiles().get("models", {}).items():
        model_name = str(profile_data.get("model", "")).strip()
        short_model_name = model_name.split("/", 1)[1] if "/" in model_name else model_name
        if (
            requested_model == profile_name
            or requested_model == model_name
            or requested_model == short_model_name
            or requested_model == profile_name.replace("_", "-")
        ):
            return profile_name
    return None


def _labels_from_metadata(metadata: Dict[str, Any] | None) -> List[str]:
    if not isinstance(metadata, dict):
        return []

    raw_labels = metadata.get("labels") or metadata.get("label") or metadata.get("classes")
    if raw_labels is None:
        return []
    if isinstance(raw_labels, str):
        return [item.strip().lower() for item in raw_labels.split(",") if item.strip()]
    if isinstance(raw_labels, (list, tuple, set)):
        return [str(item).strip().lower() for item in raw_labels if str(item).strip()]
    return []


def _preferred_profile_for_capability(capability_key: str, metadata: Dict[str, Any] | None) -> Optional[str]:
    if capability_key in {"object_detection", "object_localization"}:
        labels = _labels_from_metadata(metadata)
        if labels and any(label not in COCO_CLASSES for label in labels):
            return "yolo_world_detector"
        return "yolo11_detector"
    if capability_key == "ocr":
        return "google_vision_ocr"
    return None


def _text_parts_from_content(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return parts
    return []


def _route_query_from_messages(
    capability: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    images: Optional[Iterable[Any]] = None,
    metadata: Dict[str, Any] | None = None,
) -> str:
    capability_key = str(capability or "").strip().lower()
    category_parts = [capability_key.replace("_", " "), CAPABILITY_ROUTE_HINTS.get(capability_key, "")]

    if isinstance(metadata, dict):
        explicit = metadata.get("route_text") or metadata.get("routing_text")
        if isinstance(explicit, str) and explicit.strip():
            compact = " ".join(part.strip() for part in category_parts + [explicit] if isinstance(part, str) and part.strip())
            return compact[:6000]

    parts = list(category_parts)
    for message in messages or []:
        parts.extend(_text_parts_from_content(message.get("content")))

    if images is not None and list(images):
        parts.append("image input visual question scene camera frame")

    compact = " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    return compact[:6000]


def _model_route_text(profile_name: str, profile_data: Dict[str, Any]) -> str:
    parts = [profile_name.replace("_", " ")]
    description = profile_data.get("description")
    if isinstance(description, str):
        parts.append(description)

    for key in ("routes", "utterances", "examples", "tasks"):
        values = profile_data.get(key)
        if isinstance(values, list):
            parts.extend(str(value) for value in values if str(value).strip())
        elif isinstance(values, str):
            parts.append(values)

    return " ".join(parts)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _fallback_similarity(query: str, route_text: str) -> float:
    query_tokens = _tokenize(query)
    route_tokens = _tokenize(route_text)
    if not query_tokens or not route_tokens:
        return 0.0
    return len(query_tokens & route_tokens) / len(query_tokens | route_tokens)


def _semantic_scores(query_text: str, models: Dict[str, Any]) -> Dict[str, float]:
    route_docs = {
        profile_name: _model_route_text(profile_name, profile_data)
        for profile_name, profile_data in models.items()
        if isinstance(profile_data, dict)
    }
    if not query_text.strip() or not route_docs:
        return {}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        profile_names = list(route_docs.keys())
        docs = [query_text] + [route_docs[name] for name in profile_names]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(docs)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return {profile_name: float(score) for profile_name, score in zip(profile_names, similarities)}
    except Exception as exc:
        logger.debug(f"Semantic router falling back to token overlap: {exc}")
        return {
            profile_name: _fallback_similarity(query_text, route_text)
            for profile_name, route_text in route_docs.items()
        }


def _select_semantic_profile(query_text: str, capability_key: str) -> tuple[str, Dict[str, float]]:
    models = _model_profiles().get("models", {})
    scores = _semantic_scores(query_text, models)
    if not scores:
        return CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash"), {}

    selected_profile = max(scores, key=scores.get)
    if scores[selected_profile] < ROUTING_MIN_SCORE:
        selected_profile = CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash")
    return selected_profile, scores


def get_route_info(capability: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    models = _model_profiles().get("models", {})
    capabilities = _capability_profiles().get("capabilities", {})

    raw_capability, capability_key = _normalize_capability(capability)
    routing_mode = _routing_mode()
    capability_profile = capabilities.get(capability_key, {})
    explicit_profile = _resolve_explicit_profile(metadata)
    preferred_profile = _preferred_profile_for_capability(capability_key, metadata)
    fallback_profile = CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash")
    fallback_model = str(models.get(fallback_profile, {}).get("model", "")).strip()
    fallback_reason = None
    selected_via_fallback = False

    if explicit_profile:
        selected_profile = explicit_profile
        scoreboard = None
    elif preferred_profile:
        selected_profile = preferred_profile
        scoreboard = None
    elif routing_mode == "semantic":
        route_text = _route_query_from_messages(capability_key, metadata=metadata)
        selected_profile, scoreboard = _select_semantic_profile(route_text, capability_key)
        if not scoreboard:
            fallback_reason = "empty_scoreboard"
            selected_via_fallback = True
        elif selected_profile == fallback_profile and scoreboard.get(selected_profile, 0.0) < ROUTING_MIN_SCORE:
            fallback_reason = "below_min_score"
            selected_via_fallback = True
    elif routing_mode == "score" and capability_profile:
        selected_profile, scoreboard = _select_scored_profile(capability_key)
    else:
        selected_profile = fallback_profile
        scoreboard = None
        fallback_reason = "fixed_or_unknown_capability"
        selected_via_fallback = True

    profile_data = models.get(selected_profile, {})
    selected_model = str(profile_data.get("model", "")).strip()

    if not selected_model:
        selected_profile = "gemini_flash"
        profile_data = models.get(selected_profile, {})
        selected_model = str(profile_data.get("model", "gemini/gemini-2.0-flash")).strip()
        routing_mode = "fixed"
        fallback_profile = selected_profile
        fallback_model = selected_model
        fallback_reason = "missing_selected_model"
        selected_via_fallback = True

    if routing_mode in {"semantic", "score"} and scoreboard is not None:
        _log_scoreboard(capability_key, scoreboard, selected_profile)

    return {
        "requested_capability": raw_capability,
        "capability": capability_key,
        "task_category": capability_key,
        "selected_profile": selected_profile,
        "selected_model": selected_model,
        "provider": _provider_for_model(selected_model),
        "fallback_profile": fallback_profile if selected_via_fallback else None,
        "fallback_model": fallback_model if selected_via_fallback else None,
        "fallback_reason": fallback_reason,
        "profile_data": profile_data,
        "capability_profile": capability_profile,
        "routing_mode": routing_mode,
    }


def score(model: Dict[str, Any], capability: Dict[str, Any]) -> float:
    vision_weight = float(capability.get("vision", 0) or 0)
    coding_weight = float(capability.get("coding", 0) or 0)
    reasoning_weight = float(capability.get("reasoning", 0) or 0)
    latency_weight = float(capability.get("latency", 0) or 0)

    model_vision = float(model.get("vision", 0) or 0)
    model_coding = float(model.get("coding", 0) or 0)
    model_reasoning = float(model.get("reasoning", 0) or 0)
    model_latency = float(model.get("latency", 0) or 0)

    return (
        vision_weight * min(model_vision, float(capability.get("vision", 0) or 0))
        + coding_weight * min(model_coding, float(capability.get("coding", 0) or 0))
        + reasoning_weight * min(model_reasoning, float(capability.get("reasoning", 0) or 0))
        + latency_weight * model_latency
    )


def _select_scored_profile(capability_key: str) -> tuple[str, Dict[str, float]]:
    models = _model_profiles().get("models", {})
    capability_profile = _capability_profiles().get("capabilities", {}).get(capability_key, {})

    if not models:
        return CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash"), {}

    scores: Dict[str, float] = {}
    best_profile = None
    best_score = float("-inf")

    for profile_name, model_profile in models.items():
        model_score = score(model_profile, capability_profile)
        scores[profile_name] = model_score
        if model_score > best_score:
            best_profile = profile_name
            best_score = model_score

    if best_profile is None:
        best_profile = CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash")

    return best_profile, scores


def _log_scoreboard(capability_key: str, scoreboard: Dict[str, float], selected_profile: str) -> None:
    if not scoreboard:
        return

    lines = ["[ROUTER SCOREBOARD]", f"capability={capability_key}"]
    for profile_name in _model_profiles().get("models", {}).keys():
        if profile_name in scoreboard:
            lines.append(f"{profile_name}={scoreboard[profile_name]:.1f}")
    lines.append(f"selected={selected_profile}")
    logger.info("\n".join(lines))


def _resolve_api_key(model_name: str, metadata: Dict[str, Any] | None = None) -> str:
    if isinstance(metadata, dict):
        explicit_api_key = metadata.get("api_key") or metadata.get("explicit_api_key")
        if isinstance(explicit_api_key, str) and explicit_api_key.strip():
            return explicit_api_key.strip()

    normalized = (model_name or "").lower()
    if normalized.startswith("gemini"):
        return os.environ.get("GEMINI_API_KEY", "")
    if normalized.startswith("ollama") or normalized.startswith("llama") or normalized.startswith("llava"):
        return os.environ.get("OLLAMA_API_KEY", "")
    if normalized.startswith("groq"):
        return os.environ.get("GROQ_API_KEY", "")
    if normalized.startswith("anthropic") or normalized.startswith("claude"):
        return os.environ.get("ANTHROPIC_API_KEY", "")
    if normalized.startswith("openai") or normalized.startswith("gpt"):
        return os.environ.get("OPENAI_API_KEY", "")
    return os.environ.get("GROQ_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


def _image_to_data_uri(image: Any) -> str:
    if isinstance(image, str):
        raw = image.strip()
        return raw if raw.startswith("data:") else f"data:image/jpeg;base64,{raw}"

    if isinstance(image, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(image)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    try:
        from PIL import Image
        try:
            import numpy as np
        except ImportError:
            np = None

        if np is not None and isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                image = Image.fromarray(array)
            elif array.ndim == 3 and array.shape[2] == 3:
                image = Image.fromarray(array[:, :, ::-1].copy())
            elif array.ndim == 3 and array.shape[2] == 4:
                image = Image.fromarray(array[:, :, [2, 1, 0, 3]].copy())
            else:
                raise TypeError(f"Unsupported ndarray shape for llm_call image: {array.shape}")

        if isinstance(image, Image.Image):
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except ImportError:
        pass

    raise TypeError(f"Unsupported image type for llm_call: {type(image).__name__}")


def _merge_messages(messages: List[Dict[str, Any]], images: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    merged = [dict(message) for message in (messages or [])]
    image_items = list(images or [])
    if not image_items:
        return merged

    image_parts = [{"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}} for image in image_items]

    target_index = None
    for index in range(len(merged) - 1, -1, -1):
        if merged[index].get("role") == "user":
            target_index = index
            break

    if target_index is None:
        merged.append({"role": "user", "content": image_parts})
        return merged

    content = merged[target_index].get("content")
    if isinstance(content, list):
        merged[target_index]["content"] = list(content) + image_parts
    elif isinstance(content, str) and content.strip():
        merged[target_index]["content"] = [{"type": "text", "text": content}] + image_parts
    else:
        merged[target_index]["content"] = image_parts

    return merged


def llm_call(
    capability: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    images: Optional[Iterable[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
    task_category: Optional[str] = None,
):
    """Route a request through model-level semantic routes and call LiteLLM."""
    if not LITELLM_AVAILABLE:
        raise ImportError("litellm is not available")

    image_items = list(images or [])
    route_metadata = dict(metadata or {})
    capability = capability or task or task_category
    if not capability:
        raise ValueError("llm_call requires a capability or task category")
    messages = messages or []

    route_metadata.setdefault("route_text", _route_query_from_messages(capability, messages, image_items, route_metadata))
    route_info = get_route_info(capability, route_metadata)
    task_category = route_info["task_category"]
    selected_profile = route_info["selected_profile"]
    selected_model = route_info["selected_model"]
    provider = route_info["provider"]
    fallback_model = route_info.get("fallback_model")
    fallback_reason = route_info.get("fallback_reason")
    tool_name = route_metadata.get("tool_name") or route_metadata.get("tool")
    selected_backend = _profile_backend(route_info.get("profile_data") or {})

    if selected_backend != "litellm":
        models = _model_profiles().get("models", {})
        fallback_profile = LLM_FALLBACK_PROFILES.get(task_category, "gemini_flash")
        fallback_data = models.get(fallback_profile, {})
        fallback_model = selected_model
        fallback_reason = f"selected_backend_{selected_backend}_not_supported_by_llm_call"
        selected_profile = fallback_profile
        selected_model = str(fallback_data.get("model", "gemini/gemini-3-flash-preview")).strip()
        provider = _provider_for_model(selected_model)
        selected_backend = _profile_backend(fallback_data)

    logger.info(
        f"[ROUTER] tool_name={tool_name or 'unknown'} task_category={task_category} capability={capability} "
        f"selected_profile={selected_profile} selected_backend={selected_backend} selected_model={selected_model} provider={provider} "
        f"fallback_model={fallback_model or 'none'} fallback_reason={fallback_reason or 'none'}"
    )

    completion_kwargs: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        for key in ("temperature", "max_tokens", "top_p", "stop", "timeout", "response_format", "stream", "api_base", "base_url"):
            if key in metadata and metadata[key] is not None:
                completion_key = "api_base" if key == "base_url" else key
                completion_kwargs[completion_key] = metadata[key]

    if selected_model.startswith("ollama/"):
        completion_kwargs.setdefault("api_base", os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"))

    payload = _merge_messages(messages, image_items)
    api_key = _resolve_api_key(selected_model, metadata)

    try:
        return litellm.completion(
            model=selected_model,
            messages=payload,
            api_key=api_key,
            **completion_kwargs,
        )
    except Exception:
        logger.exception(
            f"[ROUTER] tool_name={tool_name or 'unknown'} task_category={task_category} capability={capability} "
            f"selected_profile={selected_profile} selected_backend={selected_backend} selected_model={selected_model} provider={provider} "
            f"fallback_model={fallback_model or 'none'} fallback_reason={fallback_reason or 'none'}"
        )
        raise


def vision_call(
    task: str,
    image: Any,
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    system_prompt: str = "Keep responses concise and audio-friendly.",
):
    """Route a multimodal LLM/VLM request through the central router."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return llm_call(task=task, messages=messages, images=[image], metadata=metadata)


_DETECTOR_CACHE: Dict[str, Any] = {}


def _local_model_path(model_name: str) -> str:
    name = model_name.split("/", 1)[1] if "/" in model_name else model_name
    candidate = Path(__file__).with_name(name)
    return str(candidate) if candidate.exists() else name


def _run_ultralytics_detector(
    image: Any,
    model_name: str,
    labels: Optional[List[str]],
    confidence: float,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for detector-backed routing") from exc

    cache = metadata.get("yolo_model_cache")
    if not isinstance(cache, dict):
        cache = _DETECTOR_CACHE

    model_path = _local_model_path(model_name)
    cache_key = f"model_router:{model_path}:{','.join(labels or [])}"
    if cache_key not in cache:
        cache[cache_key] = YOLO(model_path)
        if labels and "world" in model_path.lower():
            cache[cache_key].set_classes(labels)

    model = cache[cache_key]
    results = model(image, conf=confidence, verbose=False)
    wanted = {label.lower() for label in labels or []}
    detections: List[Dict[str, Any]] = []

    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", []) or []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, f"object_{class_id}"))
            if wanted and class_name.lower() not in wanted:
                continue

            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(box.conf[0]),
                "bbox": [x, y, w, h],
                "center": [x + w // 2, y + h // 2],
                "backend": "ultralytics",
                "model": model_name,
            })

    return detections


def detect_objects(
    task: str = "object_detection",
    image: Any = None,
    labels: Optional[Iterable[str]] = None,
    confidence: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Route pure object detection/localization to the configured detector backend."""
    if image is None:
        return []

    label_list = [str(label).strip().lower() for label in (labels or []) if str(label).strip()]
    route_metadata = dict(metadata or {})
    route_metadata.setdefault("labels", label_list)
    route_metadata.setdefault(
        "route_text",
        f"pure {task} for labels {', '.join(label_list) if label_list else 'common objects'}",
    )

    route_info = get_route_info(task, route_metadata)
    tool_name = route_metadata.get("tool_name") or route_metadata.get("tool")
    backend = _profile_backend(route_info.get("profile_data") or {})
    logger.info(f"[ROUTER] {_route_log_context(route_info, tool_name, 'detector_call')}")

    if backend != "local_detector":
        raise ValueError(
            f"Task {task!r} routed to backend {backend!r}; use vision_call for VLM reasoning tasks."
        )

    return _run_ultralytics_detector(
        image=image,
        model_name=route_info["selected_model"],
        labels=label_list,
        confidence=confidence,
        metadata=route_metadata,
    )


def ocr_call(
    image: Any,
    language_hints: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Route OCR/text extraction to the configured OCR backend."""
    if image is None:
        return []

    route_metadata = dict(metadata or {})
    route_metadata.setdefault("route_text", "extract readable text from image using OCR")
    route_info = get_route_info("ocr", route_metadata)
    tool_name = route_metadata.get("tool_name") or route_metadata.get("tool")
    backend = _profile_backend(route_info.get("profile_data") or {})
    logger.info(f"[ROUTER] {_route_log_context(route_info, tool_name, 'ocr_call')}")

    if backend != "ocr":
        raise ValueError(f"OCR routed to backend {backend!r}; use llm_call or vision_call for interpretation.")

    tools_dir = Path(__file__).resolve().parent.parent / "tools"
    import sys
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    from live_ocr import detect_text_google_vision

    api_key = route_metadata.get("api_key") or route_metadata.get("explicit_api_key")
    return detect_text_google_vision(
        image=image,
        api_key=api_key if isinstance(api_key, str) else None,
        language_hints=language_hints or ["en"],
    )


def _downscale_image_base64(image_base64: str, max_dim: int = 1024) -> str:
    try:
        from PIL import Image

        raw = image_base64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]

        img = Image.open(io.BytesIO(base64.b64decode(raw)))
        width, height = img.size
        if max(width, height) <= max_dim:
            return raw

        scale = max_dim / max(width, height)
        new_width, new_height = int(width * scale), int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        result = base64.b64encode(buffer.getvalue()).decode("ascii")
        logger.debug(f"Downscaled image {width}x{height} -> {new_width}x{new_height}")
        return result
    except Exception as exc:
        logger.warning(f"Image downscale failed, using original: {exc}")
        return image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64


GEMINI_IMAGE_MAX_DIM = 1024
GEMINI_LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"


class GeminiLiveSession:
    def __init__(self, api_key: str, system_instruction: str = "", model: str = None):
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.model = model or GEMINI_LIVE_MODEL
        self.session = None
        self._session_context = None
        self._client = None
        self.connected = False
        self._response_handler = None
        self._receive_task = None
        self._current_turn_text = ""
        self._current_transcript = ""
        self._turn_complete_event = asyncio.Event()
        self._send_lock = asyncio.Lock()

    async def connect(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        logger.info(f"Connecting to Gemini Live API: {self.model}")
        http_options = types.HttpOptions(api_version="v1alpha")
        self._client = genai.Client(api_key=self.api_key, http_options=http_options)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(parts=[types.Part(text=self.system_instruction)]) if self.system_instruction else None,
        )
        self._session_context = self._client.aio.live.connect(model=self.model, config=config)
        self.session = await self._session_context.__aenter__()
        self.connected = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Gemini Live session established")

    async def _receive_loop(self):
        try:
            while self.connected and self.session:
                turn = self.session.receive()
                async for response in turn:
                    if not self.connected:
                        break
                    if response.server_content:
                        sc = response.server_content
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if hasattr(part, "text") and part.text:
                                    self._current_turn_text += part.text
                        if sc.output_transcription and sc.output_transcription.text:
                            self._current_transcript += sc.output_transcription.text
                        if sc.turn_complete:
                            result = self._current_transcript.strip() or self._current_turn_text.strip()
                            self._current_turn_text = ""
                            self._current_transcript = ""
                            self._turn_complete_event.set()
                            if self._response_handler and result:
                                await self._response_handler(result, is_partial=False)
        except asyncio.CancelledError:
            logger.info("Gemini Live receive loop cancelled")
        except Exception as exc:
            if self.connected:
                logger.error(f"Gemini Live receive loop error: {exc}")
            self.connected = False

    def set_response_handler(self, handler):
        self._response_handler = handler

    async def send_image_query(self, image_base64: str, query_text: str, mime_type: str = "image/jpeg") -> str:
        if not self.connected or not self.session:
            raise ConnectionError("Gemini Live session not connected")

        async with self._send_lock:
            image_base64 = _downscale_image_base64(image_base64, GEMINI_IMAGE_MAX_DIM)
            self._current_turn_text = ""
            self._current_transcript = ""
            self._turn_complete_event.clear()
            await self.session.send_client_content(
                turns=[{"role": "user", "parts": [{"inline_data": {"mime_type": mime_type, "data": image_base64}}, {"text": query_text}]}],
                turn_complete=True,
            )
            try:
                await asyncio.wait_for(self._turn_complete_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                return self._current_transcript.strip() or self._current_turn_text.strip() or "Response timed out"
            return self._current_transcript.strip() or self._current_turn_text.strip()

    async def send_followup(self, text: str) -> str:
        if not self.connected or not self.session:
            raise ConnectionError("Gemini Live session not connected")

        async with self._send_lock:
            self._current_turn_text = ""
            self._current_transcript = ""
            self._turn_complete_event.clear()
            await self.session.send_client_content(turns=[{"role": "user", "parts": [{"text": text}]}], turn_complete=True)
            try:
                await asyncio.wait_for(self._turn_complete_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                return self._current_transcript.strip() or self._current_turn_text.strip() or "Response timed out"
            return self._current_transcript.strip() or self._current_turn_text.strip()

    async def close(self):
        self.connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning(f"Error closing Gemini Live session context: {exc}")
            self._session_context = None
        self.session = None
        logger.info("Gemini Live session closed")


class GeminiLiveManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sessions: dict[str, GeminiLiveSession] = {}
        self._query_tasks: dict[str, asyncio.Task] = {}
        self._paused: dict[str, bool] = {}

    async def start_session(self, client_id: str, system_instruction: str, response_handler) -> GeminiLiveSession:
        await self.stop_session(client_id)
        brevity_prefix = (
            "Be extremely concise. Respond in 1-2 short sentences max. "
            "Your responses will be spoken aloud, so keep them brief and direct. "
            "You will receive a series of images from a live camera feed. "
            "Each image is a new independent frame — only describe what you see in the CURRENT image. "
            "Never compare to or reference previous images. Treat each as standalone. "
        )
        session = GeminiLiveSession(self.api_key, brevity_prefix + (system_instruction or ""))
        session.set_response_handler(response_handler)
        await session.connect()
        self.sessions[client_id] = session
        return session

    async def run_query_loop(self, client_id: str, query_text: str, get_current_frame, interval_seconds: float = 5.0):
        session = self.sessions.get(client_id)
        if not session:
            logger.error(f"No Gemini Live session for {client_id}")
            return

        try:
            while session.connected and client_id in self.sessions:
                if self._paused.get(client_id, False):
                    await asyncio.sleep(0.5)
                    continue

                try:
                    frame_base64, _ = get_current_frame()
                    if frame_base64:
                        await session.send_image_query(frame_base64, query_text)
                except ConnectionError:
                    break
                except Exception as exc:
                    logger.error(f"Error in live query loop for {client_id}: {exc}")

                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info(f"Live query loop cancelled for {client_id}")

    def pause_query_loop(self, client_id: str):
        self._paused[client_id] = True

    def resume_query_loop(self, client_id: str):
        self._paused[client_id] = False

    async def send_followup(self, client_id: str, text: str) -> str:
        session = self.sessions.get(client_id)
        if not session or not session.connected:
            raise ConnectionError(f"No active Gemini Live session for {client_id}")
        return await session.send_followup(text)

    async def stop_session(self, client_id: str):
        self._paused.pop(client_id, None)
        if client_id in self._query_tasks:
            self._query_tasks[client_id].cancel()
            try:
                await self._query_tasks[client_id]
            except asyncio.CancelledError:
                pass
            del self._query_tasks[client_id]
        if client_id in self.sessions:
            await self.sessions[client_id].close()
            del self.sessions[client_id]

    async def stop_all(self):
        for client_id in list(self.sessions.keys()):
            await self.stop_session(client_id)


__all__ = [
    "llm_call",
    "vision_call",
    "detect_objects",
    "ocr_call",
    "get_route_info",
    "TASK_CATEGORIES",
    "GeminiLiveSession",
    "GeminiLiveManager",
    "GEMINI_IMAGE_MAX_DIM",
    "GEMINI_LIVE_MODEL",
]
