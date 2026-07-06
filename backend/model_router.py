"""Atomic capability execution and fixed infrastructure LLM calls for ProgramAT."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.request
import urllib.error
from contextvars import ContextVar
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
    "Use useful previous-stage detection results as additional context. "
    "A missing, failed, or uncertain detection is not evidence that the target is absent; "
    "inspect the image independently when prior-stage context is uncertain. "
    "Give concise spoken instructions "
    "in at most 2 short sentences. Do not use a numbered list, introduction, "
    "explanation, conversational filler, safety disclaimer, 'keep an eye out', "
    "or 'don't hesitate to ask'."
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
_IMPLEMENTATION_MODEL_CACHE: Dict[str, Any] = {}
STREAMING_EXECUTION_CONTEXT: ContextVar[bool] = ContextVar(
    "streaming_execution", default=False
)
TOOL_EXECUTION_IMAGES: ContextVar[Optional[List[Any]]] = ContextVar(
    "tool_execution_images", default=None
)


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
    model_name = profile.model_name or "yolo11n.pt"
    model = cache.get(model_name) or YOLO(model_name)
    cache[model_name] = model
    target_labels = _target_labels(metadata)
    logger.info("[Target Grounding] object_detection_localization target_labels=%s", target_labels)
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
        text = "The detector could not confidently localize the requested object."
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
    response = call_model(profile.model, call_messages, images=images, metadata=metadata)
    text = _response_text(response)
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
    elif policy.get("specialized"):
        logger.info(
            "[Execution Policy] routing disabled -> preserving specialized implementation=%s",
            policy["candidates"][0],
        )
        candidates = policy["candidates"]
        evaluator_name = None
    else:
        logger.info("[Execution Policy] routing disabled -> using default model for all LLM stages")
        candidates = [global_config["default_llm_when_routing_disabled"]]
        evaluator_name = None

    call_metadata = dict(metadata or {})
    call_metadata.setdefault("model_cache", _IMPLEMENTATION_MODEL_CACHE)
    call_metadata["capability"] = declared
    streaming_log = bool(
        call_metadata.get("streaming") or STREAMING_EXECUTION_CONTEXT.get()
    )
    if task and "task_text" not in call_metadata:
        call_metadata["task_text"] = task
    call_messages = list(messages or [])
    target_labels = _target_labels(call_metadata)
    grounding_context = [goal, task, call_metadata.get("goal"), call_metadata.get("task_text")]
    grounding_context.extend(message.get("content") for message in call_messages if isinstance(message, dict))
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
    output = None
    implementation = ""
    best_output = None
    best_implementation = ""
    selected = False
    for index, candidate in enumerate(candidates):
        if streaming_log:
            logger.info(
                "[Streaming] implementation candidate=%s capability=%s",
                candidate,
                declared,
            )
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
                previous_text = _previous_stage_text(call_metadata, previous_artifact)
                evaluator_metadata = {
                    "temperature": 0,
                    "max_tokens": 3,
                    "capability": declared,
                    "evaluator": True,
                }
                logger.info("[Evaluator] image_included=false")
                evaluation = evaluator(
                    evaluator_profile,
                    [{
                        "role": "user",
                        "content": EVALUATION_PROMPT.format(
                            capability=declared,
                            goal=goal or call_metadata.get("goal") or call_metadata.get("task_text") or "",
                            previous_text=previous_text,
                            response=response_text,
                        ),
                    }],
                    [],
                    evaluator_metadata,
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
            logger.info("[Evaluator] decision=%s", decision)
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    reason_output = evaluator(
                        evaluator_profile,
                        [{
                            "role": "user",
                            "content": EVALUATION_REASON_PROMPT.format(
                                capability=declared,
                                goal=goal or call_metadata.get("goal") or call_metadata.get("task_text") or "",
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
                    logger.debug(
                        "[Evaluator] decision=%s reason=%s",
                        decision,
                        json.dumps(reason, ensure_ascii=False),
                    )
                except Exception as exc:
                    logger.debug(
                        "[Evaluator] decision=%s reason=%s",
                        decision,
                        json.dumps(f"reason unavailable: {exc}"),
                    )
            if streaming_log:
                logger.info(
                    "[Streaming] evaluator=%s capability=%s decision=%s",
                    evaluator_name,
                    declared,
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
            fallback_text = "The previous stage could not produce a reliable result."
            output = ImplementationResult(
                _simple_response(fallback_text),
                {"text": fallback_text, "accepted": False, "error": "stage_failed"},
            )
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
    "LEGACY_CAPABILITY_ALIASES", "NAVIGATION_SYSTEM_PROMPT", "TOOL_EXECUTION_IMAGES",
    "EVALUATION_PROMPT", "EVALUATION_REASON_PROMPT", "STREAMING_RESPONSE_PROMPT",
    "ImplementationProfile", "ImplementationResult", "ExecutionPolicyError",
    "IMPLEMENTATION_EXECUTORS",
    "normalize_capability_name", "load_capability_descriptions", "load_capability_profiles",
    "load_execution_policies", "load_implementation_profiles", "load_global_execution_config",
    "validate_execution_configuration",
    "system_llm_call", "single_stage_llm_call", "copilot_llm_call",
]
