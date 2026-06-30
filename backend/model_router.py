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
    "Use previous-stage detection results as the primary source of truth. "
    "Use the image only to refine navigation. Give concise spoken instructions "
    "in at most 2 short sentences. Do not use a numbered list, introduction, "
    "explanation, conversational filler, safety disclaimer, 'keep an eye out', "
    "or 'don't hesitate to ask'."
)
EVALUATION_PROMPT = """Evaluate the previous response for the requested capability.

Consider:
- Did it answer the requested capability?
- Is it actionable?
- Is it specific enough for a blind or low-vision user?
- Is important information missing?
- Does it contradict previous-stage artifacts?

Answer only YES or NO.

Capability: {capability}
Goal: {goal}
Previous-stage artifact: {artifact}
Previous response: {response}

Is the previous response sufficient to accomplish the requested capability for a blind or low-vision user?"""
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


def load_execution_policies(path: Path = EXECUTION_POLICY_PATH) -> Dict[str, Dict[str, Any]]:
    config = _load_yaml(path)
    config.pop("global", None)
    config.pop("implementations", None)
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
    enabled = raw.get("model_routing_enabled")
    if not isinstance(enabled, bool):
        raise ExecutionPolicyError("global.model_routing_enabled must be true or false")

    def implementation_name(field: str) -> str:
        value = raw.get(field)
        if not isinstance(value, dict) or not str(value.get("implementation") or "").strip():
            raise ExecutionPolicyError(f"global.{field}.implementation is required")
        return str(value["implementation"]).strip()

    return {
        "model_routing_enabled": enabled,
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
        "[Execution Policy] model_routing_enabled=%s",
        str(config["model_routing_enabled"]).lower(),
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
    global_config = load_global_execution_config()
    _log_global_execution_config(global_config, implementations)
    if declared not in policies:
        raise ExecutionPolicyError(
            f"Unknown capability {declared!r}; supported capabilities are: {sorted(policies)}"
        )
    policy = policies[declared]
    if global_config["model_routing_enabled"] or policy.get("specialized"):
        candidates = policy["candidates"]
        evaluator_name = policy.get("evaluator")
    else:
        candidates = [global_config["default_llm_when_routing_disabled"]]
        evaluator_name = None

    call_metadata = dict(metadata or {})
    call_metadata.setdefault("model_cache", _IMPLEMENTATION_MODEL_CACHE)
    call_metadata["capability"] = declared
    if task and "task_text" not in call_metadata:
        call_metadata["task_text"] = task
    call_messages = list(messages or [])
    if declared == "navigation":
        call_messages.insert(0, {"role": "system", "content": NAVIGATION_SYSTEM_PROMPT})
    previous_artifact = call_metadata.get("previous_stage_artifact")
    if previous_artifact is not None:
        call_messages.append({
            "role": "user",
            "content": "Previous-stage artifact (primary source of truth): "
            + json.dumps(previous_artifact, ensure_ascii=False, default=str),
        })
    if goal:
        call_metadata["goal"] = str(goal)
        call_messages.append({"role": "user", "content": str(goal)})
    call_images = list(images or [])
    output = None
    implementation = ""
    best_output = None
    best_implementation = ""
    selected = False
    for index, candidate in enumerate(candidates):
        logger.info("[Execution Policy] capability=%s trying=%s", declared, candidate)
        try:
            profile = implementations[candidate]
            executor = IMPLEMENTATION_EXECUTORS.get(profile.kind)
            if executor is None:
                raise ExecutionPolicyError(
                    f"No executor for implementation kind {profile.kind!r}"
                )
            candidate_output = executor(profile, call_messages, call_images, call_metadata)
        except Exception as exc:
            logger.warning(
                "[Execution Policy] capability=%s implementation=%s failed=%s",
                declared,
                candidate,
                exc,
            )
            continue

        response_text = _response_text(candidate_output.response)
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

        if evaluator_name and index < len(candidates) - 1:
            try:
                evaluator_profile = implementations[evaluator_name]
                evaluator = IMPLEMENTATION_EXECUTORS.get(evaluator_profile.kind)
                if evaluator is None:
                    raise ExecutionPolicyError(
                        f"No executor for evaluator implementation kind {evaluator_profile.kind!r}"
                    )
                evaluation = evaluator(
                    evaluator_profile,
                    [{
                        "role": "user",
                        "content": EVALUATION_PROMPT.format(
                            capability=declared,
                            goal=goal or call_metadata.get("goal") or call_metadata.get("task_text") or "",
                            artifact=json.dumps(previous_artifact, ensure_ascii=False, default=str),
                            response=response_text,
                        ),
                    }],
                    [],
                    {**call_metadata, "temperature": 0, "max_tokens": 3},
                )
                raw_decision = _response_text(evaluation.response).strip().upper()
                if raw_decision not in {"YES", "NO"}:
                    raise ValueError("evaluator did not return YES or NO")
                decision = raw_decision
            except Exception as exc:
                logger.warning(
                    "[Execution Policy] capability=%s evaluator=%s decision=FAILED error=%s",
                    declared,
                    evaluator_name,
                    exc,
                )
                continue

            logger.info(
                "[Execution Policy] capability=%s evaluator=%s decision=%s",
                declared,
                evaluator_name,
                decision,
            )
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
            fallback_text = "Sorry, I couldn't generate guidance from this image."
            output = ImplementationResult(_simple_response(fallback_text), {"text": fallback_text})
            implementation = "fallback"
            logger.warning("[Execution Policy] capability=%s fallback=none", declared)
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
    "BACKEND_DIR", "CAPABILITY_PROFILES_PATH", "EXECUTION_POLICY_PATH",
    "LEGACY_CAPABILITY_ALIASES", "NAVIGATION_SYSTEM_PROMPT",
    "EVALUATION_PROMPT",
    "ImplementationProfile", "ImplementationResult", "ExecutionPolicyError",
    "IMPLEMENTATION_EXECUTORS",
    "normalize_capability_name", "load_capability_descriptions", "load_capability_profiles",
    "load_execution_policies", "load_implementation_profiles", "load_global_execution_config",
    "validate_execution_configuration",
    "system_llm_call", "copilot_llm_call",
]
