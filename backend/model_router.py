"""Atomic capability execution and fixed infrastructure LLM calls for ProgramAT."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from litellm_utils import call_model, extract_text


logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PROFILES_PATH = BACKEND_DIR / "model_profiles.yaml"
CAPABILITY_PROFILES_PATH = BACKEND_DIR / "capability_profiles.yaml"
EXECUTION_POLICY_PATH = BACKEND_DIR / "execution_policy.yaml"
SYSTEM_MODEL = os.environ.get("SYSTEM_LLM_MODEL", "groq/llama-3.1-8b-instant")
DEFAULT_FALLBACK_CAPABILITY = "general_reasoning"
LEGACY_CAPABILITY_ALIASES = {
    "object_detection": "object_detection_localization",
}
NAVIGATION_SYSTEM_PROMPT = (
    "You provide navigation guidance for a blind or low-vision user. "
    "Use previous-stage detection results as the primary source of truth. "
    "Give concise spoken instructions in at most 2-3 short sentences. "
    "Do not use a numbered list, conversational filler, 'keep an eye out', "
    "or 'don't hesitate to ask'."
)
_IMPLEMENTATION_MODEL_CACHE: Dict[str, Any] = {}


class ExecutionPolicyError(ValueError):
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


def load_execution_policies(path: Path = EXECUTION_POLICY_PATH) -> Dict[str, List[str]]:
    policies = {}
    for capability, raw in _load_yaml(path).items():
        if not isinstance(raw, dict) or not isinstance(raw.get("implementations"), list):
            raise ExecutionPolicyError(f"Capability {capability!r} must define an implementations list")
        names = [str(value).strip() for value in raw["implementations"] if str(value).strip()]
        if not names:
            raise ExecutionPolicyError(f"Capability {capability!r} has no implementations")
        policies[str(capability).strip()] = names
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


def load_implementation_profiles(path: Path = MODEL_PROFILES_PATH) -> Dict[str, ImplementationProfile]:
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


def validate_execution_configuration(
    policies: Optional[Mapping[str, Sequence[str]]] = None,
    implementations: Optional[Mapping[str, ImplementationProfile]] = None,
) -> None:
    taxonomy = set(load_capability_profiles())
    policies = policies or load_execution_policies()
    implementations = implementations or load_implementation_profiles()

    policy_names = set(policies)
    if policy_names != taxonomy:
        missing = taxonomy - policy_names
        extra = policy_names - taxonomy
        raise ExecutionPolicyError(
            "Capability taxonomy mismatch between capability_profiles.yaml and "
            "execution_policy.yaml; "
            f"missing_policies={sorted(missing)}, unknown_policies={sorted(extra)}"
        )

    unknown = {values[0] for values in policies.values() if values[0] not in implementations}
    if unknown:
        raise ExecutionPolicyError(f"Unknown first implementations: {', '.join(sorted(unknown))}")


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


def _google_vision_executor(profile, messages, images, metadata) -> ImplementationResult:
    image_items = list(images or [])
    if not image_items:
        return ImplementationResult(_simple_response("No image was provided."), {"text": ""})
    api_key = str(metadata.get("api_key") or os.environ.get("GOOGLE_VISION_API_KEY") or os.environ.get("GEMINI_API_KEY") or "")
    if not api_key:
        raise RuntimeError("Google Vision API key is not configured")
    body = json.dumps({"requests": [{"image": {"content": base64.b64encode(_image_bytes(image_items[0])).decode("ascii")}, "features": [{"type": "TEXT_DETECTION"}]}]}).encode("utf-8")
    url = "https://vision.googleapis.com/v1/images:annotate?" + urllib.parse.urlencode({"key": api_key})
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=metadata.get("timeout", 30)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    first = (payload.get("responses") or [{}])[0]
    if first.get("error"):
        raise RuntimeError(first["error"].get("message", "Google Vision OCR failed"))
    annotations = first.get("textAnnotations") or []
    text = str(annotations[0].get("description", "")).strip() if annotations else ""
    return ImplementationResult(_simple_response(text), {"text": text})


def _location(bbox: Sequence[float], width: int, height: int) -> str:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    horizontal = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "center"
    vertical = "top" if y < height / 3 else "bottom" if y > 2 * height / 3 else "middle"
    return horizontal if vertical == "middle" else f"{vertical} {horizontal}"


def _yolo_executor(profile, messages, images, metadata) -> ImplementationResult:
    image_items = list(images or [])
    if not image_items:
        return ImplementationResult(_simple_response("No image was provided."), {"detections": []})
    from PIL import Image
    from ultralytics import YOLO
    image = Image.open(io.BytesIO(_image_bytes(image_items[0]))).convert("RGB")
    cache = metadata.get("model_cache") if isinstance(metadata.get("model_cache"), dict) else {}
    model_name = profile.model_name or "yolo11n.pt"
    model = cache.get(model_name) or YOLO(model_name)
    cache[model_name] = model
    detections = []
    for result in model(image, verbose=False):
        for box in result.boxes:
            bbox = [float(value) for value in box.xyxy[0].tolist()]
            detections.append({"label": str(result.names[int(box.cls[0])]), "bbox": bbox, "location": _location(bbox, image.width, image.height)})
    text = "; ".join(f"{item['label']} at {item['location']}" for item in detections) or "No requested object was detected."
    return ImplementationResult(_simple_response(text), {"detections": detections})


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
    "google_vision": _google_vision_executor,
    "yolo": _yolo_executor,
    "groundingdino": _groundingdino_executor,
}


def system_llm_call(messages=None, images=None, metadata=None):
    """Call the fixed infrastructure model without execution-policy routing."""
    logger.info("[System LLM] model=%s", SYSTEM_MODEL)
    return call_model(SYSTEM_MODEL, messages or [], images=images, metadata=metadata)


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
    policies = load_execution_policies()
    implementations = load_implementation_profiles()
    if declared not in policies:
        raise ExecutionPolicyError(
            f"Unknown capability {declared!r}; supported capabilities are: {sorted(policies)}"
        )
    implementation = policies[declared][0]
    profile = implementations[implementation]
    executor = IMPLEMENTATION_EXECUTORS.get(profile.kind)
    if executor is None:
        raise ExecutionPolicyError(f"No executor for implementation kind {profile.kind!r}")

    call_metadata = dict(metadata or {})
    call_metadata.setdefault("model_cache", _IMPLEMENTATION_MODEL_CACHE)
    call_metadata["capability"] = declared
    if task and "task_text" not in call_metadata:
        call_metadata["task_text"] = task
    call_messages = list(messages or [])
    if declared == "navigation":
        call_messages.insert(0, {"role": "system", "content": NAVIGATION_SYSTEM_PROMPT})
    if goal:
        call_metadata["goal"] = str(goal)
        call_messages.append({"role": "user", "content": str(goal)})
    logger.info("[Execution Policy] capability=%s implementation=%s", declared, implementation)
    output = executor(profile, call_messages, list(images or []), call_metadata)
    artifact = output.artifact
    if artifact is None:
        artifact = {"text": _response_text(output.response)}
    return {
        "response": _response_text(output.response),
        "artifact": artifact,
        "implementation": implementation,
        "capability": declared,
    }


__all__ = [
    "BACKEND_DIR", "MODEL_PROFILES_PATH", "CAPABILITY_PROFILES_PATH", "EXECUTION_POLICY_PATH",
    "SYSTEM_MODEL", "LEGACY_CAPABILITY_ALIASES", "NAVIGATION_SYSTEM_PROMPT",
    "ImplementationProfile", "ImplementationResult", "ExecutionPolicyError",
    "IMPLEMENTATION_EXECUTORS",
    "normalize_capability_name", "load_capability_descriptions", "load_capability_profiles",
    "load_execution_policies", "load_implementation_profiles", "validate_execution_configuration",
    "system_llm_call", "copilot_llm_call",
]
