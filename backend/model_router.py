"""Semantic capability router and simple LLM call entrypoints for ProgramAT."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from litellm_utils import call_model


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PROFILES_PATH = BACKEND_DIR / "model_profiles.yaml"
CAPABILITY_PROFILES_PATH = BACKEND_DIR / "capability_profiles.yaml"
SYSTEM_MODEL = os.environ.get("SYSTEM_LLM_MODEL", "gemini/gemini-2.0-flash-preview")
DEFAULT_FALLBACK_CAPABILITY = "general_reasoning"

STRUCTURED_ARTIFACTS_BY_CAPABILITY = {
    "object_detection": "target bounding box and object coordinates",
    "ocr": "extracted text",
    "spatial_relationship": "target location and relative position",
    "navigation": "navigation waypoint",
    "camera_motion": "camera framing adjustment instructions",
    "map_web": "structured visual region or layout element",
}

HOLISTIC_TASK_KEYWORDS = (
    "describe",
    "explain",
    "summarize",
    "rate",
    "evaluate",
)

TARGET_REDUCTION_KEYWORDS = (
    "find",
    "locate",
    "guide",
    "navigate",
    "entrance",
    "elevator",
    "door handle",
    "handle",
    "bus",
    "target",
    "align",
    "where",
)

CAMERA_GUIDANCE_KEYWORDS = (
    "camera",
    "center",
    "framing",
    "frame",
    "zoom",
    "left",
    "right",
    "up",
    "down",
    "closer",
    "farther",
    "aim",
    "align",
)

WAYFINDING_KEYWORDS = (
    "walk",
    "route",
    "wayfinding",
    "reach",
    "entrance",
    "exit",
    "go to",
)

TEXT_ANALYSIS_KEYWORDS = (
    "summarize",
    "summary",
    "explain",
    "interpret",
    "understand",
    "analyze",
)

PENALTY_PER_PIPELINE_STAGE = 0.08
MIN_PIPELINE_SCORE = 0.35
PIPELINE_REWRITE_MARGIN = 0.05

INFORMATION_REDUCTION_SCORES = {
    "none": 0.0,
    "minor": 0.25,
    "moderate": 0.6,
    "significant": 1.0,
}

USEFUL_STAGE_TRANSITIONS = {
    ("object_detection", "camera_motion"),
    ("object_detection", "spatial_relationship"),
    ("object_detection", "navigation"),
    ("object_detection", "general_reasoning"),
    ("object_detection", "map_web"),
    ("ocr", "general_reasoning"),
    ("ocr", "map_web"),
    ("map_web", "general_reasoning"),
    ("spatial_relationship", "navigation"),
    ("spatial_relationship", "general_reasoning"),
    ("camera_motion", "navigation"),
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    type: str
    latency_ms: float
    source: str
    capabilities: Dict[str, float]
    latency: float = 0.5
    model: str = ""


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_capability_descriptions(path: Path = CAPABILITY_PROFILES_PATH) -> Dict[str, List[str]]:
    profiles = load_capability_profiles(path)
    descriptions: Dict[str, List[str]] = {}
    for capability, profile in profiles.items():
        values = [profile.get("description", "")]
        values.extend(profile.get("include_examples", []))
        notes = str(profile.get("notes", "")).strip()
        if notes:
            values.append(notes)
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned:
            raise ValueError(f"Capability {capability!r} must provide description and/or include_examples")
        descriptions[capability] = cleaned
    return descriptions


def load_capability_profiles(path: Path = CAPABILITY_PROFILES_PATH) -> Dict[str, Dict[str, List[str] | str]]:
    data = _load_yaml(path)
    capabilities = data.get("capabilities", {})
    if not isinstance(capabilities, dict) or not capabilities:
        raise ValueError(f"No capabilities configured in {path}")

    profiles: Dict[str, Dict[str, List[str] | str]] = {}
    for capability, raw in capabilities.items():
        capability_name = str(capability)
        if isinstance(raw, list):
            values = [str(value).strip() for value in raw if str(value).strip()]
            if not values:
                raise ValueError(f"Capability {capability_name!r} must contain a non-empty list of descriptions")
            profiles[capability_name] = {
                "description": values[0],
                "include_examples": values[1:],
                "exclude_examples": [],
                "notes": "",
            }
            continue

        if not isinstance(raw, dict):
            raise ValueError(
                f"Capability {capability_name!r} must be a list or mapping with description/include_examples/exclude_examples"
            )

        description = str(raw.get("description", "")).strip()
        include_examples = raw.get("include_examples", []) or []
        exclude_examples = raw.get("exclude_examples", []) or []
        notes = raw.get("notes", "")
        if not isinstance(include_examples, list) or not isinstance(exclude_examples, list):
            raise ValueError(
                f"Capability {capability_name!r} include_examples/exclude_examples must be lists"
            )
        profiles[capability_name] = {
            "description": description,
            "include_examples": [str(value).strip() for value in include_examples if str(value).strip()],
            "exclude_examples": [str(value).strip() for value in exclude_examples if str(value).strip()],
            "notes": str(notes).strip(),
        }
    return profiles


def load_model_profiles(path: Path = MODEL_PROFILES_PATH) -> List[ModelProfile]:
    data = _load_yaml(path)
    models = data.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No models configured in {path}")

    profiles: List[ModelProfile] = []
    for name, raw_profile in models.items():
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Model {name!r} must be a mapping")
        capabilities = raw_profile.get("strengths", raw_profile.get("capabilities", {}))
        if not isinstance(capabilities, dict):
            raise ValueError(f"Model {name!r} capabilities/strengths must be a mapping")
        latency_ms = float(raw_profile.get("latency_ms", 1000))
        latency = raw_profile.get("latency")
        if latency is None:
            latency = max(0.0, min(1.0, 1.0 - latency_ms / 5000.0))
        profiles.append(ModelProfile(
            name=str(name),
            type=str(raw_profile.get("type", "general_vlm")),
            latency_ms=latency_ms,
            source=str(raw_profile.get("source", "unknown")),
            capabilities={str(key): float(value or 0.0) for key, value in capabilities.items()},
            latency=float(max(0.0, min(1.0, float(latency)))),
            model=str(raw_profile.get("model", "") or ""),
        ))
    return profiles


def _task_strength(model: ModelProfile, capability_name: str) -> float:
    key = str(capability_name or "").strip().lower()
    return float(model.capabilities.get(key, 0.0))


def _safe_weight(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_routing_analysis(
    routing_analysis: Optional[Dict[str, Any]],
    supported_capabilities: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(routing_analysis, dict):
        return None

    if supported_capabilities is None:
        supported_capabilities = load_capability_descriptions().keys()
    canonical_capabilities = {
        str(name).strip().lower(): str(name).strip().lower()
        for name in supported_capabilities
        if str(name).strip()
    }
    if not canonical_capabilities:
        return None

    tasks = routing_analysis.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None

    normalized_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        raw_name = str(task.get("name", "")).strip().lower()
        key = canonical_capabilities.get(raw_name)
        if key is None:
            continue
        weight = max(0.0, _safe_weight(task.get("weight"), 0.0))
        if weight <= 0.0:
            continue
        normalized_tasks.append({
            "name": key,
            "weight": weight,
            "reason": str(task.get("reason", "")).strip(),
        })

    if not normalized_tasks:
        return None

    total = sum(task["weight"] for task in normalized_tasks)
    if total <= 0.0:
        return None
    for task in normalized_tasks:
        task["weight"] = task["weight"] / total

    def _normalize_sensitivity(raw: Any, default_level: str, default_weight: float) -> Dict[str, Any]:
        level = default_level
        weight = default_weight
        reason = ""
        if isinstance(raw, dict):
            raw_level = str(raw.get("level", default_level)).strip().lower()
            if raw_level in {"low", "medium", "high"}:
                level = raw_level
            weight = max(0.0, min(1.0, _safe_weight(raw.get("weight"), default_weight)))
            reason = str(raw.get("reason", "")).strip()
        return {"level": level, "weight": weight, "reason": reason}

    return {
        "tasks": normalized_tasks,
        "latency_sensitivity": _normalize_sensitivity(routing_analysis.get("latency_sensitivity"), "medium", 0.5),
    }


def _canonical_capability_map(supported_capabilities: Optional[Iterable[str]] = None) -> Dict[str, str]:
    if supported_capabilities is None:
        supported_capabilities = load_capability_descriptions().keys()
    return {
        str(name).strip().lower(): str(name).strip().lower()
        for name in supported_capabilities
        if str(name).strip()
    }


def normalize_pipeline_analysis(
    pipeline_analysis: Optional[Dict[str, Any]],
    supported_capabilities: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    default = {
        "should_chain": False,
        "reason": "No valid pipeline_analysis was provided; defaulting to a single model.",
        "stages": [],
        "artifact_value_score": 0.0,
        "information_reduction": "none",
        "stage_count": 0,
        "artifact_gain": 0.0,
        "complexity_penalty": 0.0,
        "pipeline_score": 0.0,
    }
    if not isinstance(pipeline_analysis, dict):
        return default

    should_chain = bool(pipeline_analysis.get("should_chain"))
    reason = str(pipeline_analysis.get("reason", "")).strip()
    canonical_capabilities = _canonical_capability_map(supported_capabilities)
    normalized_stages = []
    raw_stages = pipeline_analysis.get("stages", [])

    if isinstance(raw_stages, list):
        for raw_stage in raw_stages:
            if not isinstance(raw_stage, dict):
                continue
            raw_capability = str(raw_stage.get("capability", "")).strip().lower()
            capability = canonical_capabilities.get(raw_capability)
            if capability is None:
                continue
            normalized_stages.append({
                "capability": capability,
                "purpose": str(raw_stage.get("purpose", "")).strip(),
                "input": str(raw_stage.get("input", "")).strip(),
                "output": str(raw_stage.get("output", "")).strip(),
                "preferred_model_type": str(raw_stage.get("preferred_model_type", "")).strip(),
            })

    if not should_chain:
        return _with_pipeline_score({
            "should_chain": False,
            "reason": reason or "No structured intermediate artifact is expected to reduce later visual scope or reasoning burden.",
            "stages": [],
            "artifact_value_score": max(0.0, min(1.0, _safe_weight(pipeline_analysis.get("artifact_value_score"), 0.0))),
            "information_reduction": str(pipeline_analysis.get("information_reduction", "none")).strip() or "none",
        })

    if len(normalized_stages) < 2:
        return _with_pipeline_score({
            "should_chain": False,
            "reason": reason or "Pipeline request did not include at least two valid stages; defaulting to a single model.",
            "stages": [],
            "artifact_value_score": 0.0,
            "information_reduction": "none",
        })

    return _with_pipeline_score({
        "should_chain": True,
        "reason": reason or "A specialized intermediate result is expected to reduce later model work.",
        "stages": normalized_stages,
        "artifact_value_score": max(0.0, min(1.0, _safe_weight(pipeline_analysis.get("artifact_value_score"), 0.75))),
        "information_reduction": str(pipeline_analysis.get("information_reduction", "significant")).strip() or "significant",
    })


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _pipeline_score_details(pipeline_analysis: Dict[str, Any]) -> Dict[str, float | int]:
    stages = pipeline_analysis.get("stages", [])
    stage_count = len(stages) if isinstance(stages, list) else 0
    if not pipeline_analysis.get("should_chain") or stage_count < 2:
        return {
            "stage_count": stage_count,
            "artifact_gain": 0.0,
            "complexity_penalty": 0.0,
            "pipeline_score": 0.0,
            "useful_transition_count": 0,
        }

    information_reduction = str(pipeline_analysis.get("information_reduction", "none")).strip().lower()
    reduction_score = INFORMATION_REDUCTION_SCORES.get(information_reduction, 0.0)
    artifact_value = max(0.0, min(1.0, _safe_weight(pipeline_analysis.get("artifact_value_score"), 0.0)))

    transitions = []
    for index in range(stage_count - 1):
        current = str(stages[index].get("capability", "")).strip().lower()
        next_stage = str(stages[index + 1].get("capability", "")).strip().lower()
        if current and next_stage:
            transitions.append((current, next_stage))

    useful_transition_count = sum(1 for transition in transitions if transition in USEFUL_STAGE_TRANSITIONS)
    useful_transition_ratio = useful_transition_count / max(1, len(transitions))
    useful_transition_bonus = min(0.25, useful_transition_count * 0.08)
    artifact_gain = (
        0.50 * artifact_value
        + 0.25 * reduction_score
        + 0.15 * useful_transition_ratio
        + useful_transition_bonus
    )
    complexity_penalty = max(0, stage_count - 1) * PENALTY_PER_PIPELINE_STAGE
    pipeline_score = artifact_gain - complexity_penalty
    return {
        "stage_count": stage_count,
        "artifact_gain": round(artifact_gain, 4),
        "complexity_penalty": round(complexity_penalty, 4),
        "pipeline_score": round(pipeline_score, 4),
        "useful_transition_count": useful_transition_count,
    }


def _with_pipeline_score(pipeline_analysis: Dict[str, Any]) -> Dict[str, Any]:
    scored = dict(pipeline_analysis)
    scored.update(_pipeline_score_details(scored))
    if scored.get("should_chain") and scored.get("pipeline_score", 0.0) < MIN_PIPELINE_SCORE:
        return {
            "should_chain": False,
            "reason": (
                "Pipeline artifact gain did not exceed complexity penalty; "
                "using the minimum useful single-model plan."
            ),
            "stages": [],
            "artifact_value_score": scored.get("artifact_value_score", 0.0),
            "information_reduction": scored.get("information_reduction", "none"),
            **_pipeline_score_details({"should_chain": False, "stages": []}),
        }
    return scored


def _is_holistic_without_target_reduction(task_text: str) -> bool:
    lowered = task_text.lower()
    return _contains_any(lowered, HOLISTIC_TASK_KEYWORDS) and not _contains_any(lowered, TARGET_REDUCTION_KEYWORDS)


def _stage(
    capability: str,
    purpose: str,
    input_text: str,
    output: str,
    preferred_model_type: str,
) -> Dict[str, str]:
    return {
        "capability": capability,
        "purpose": purpose,
        "input": input_text,
        "output": output,
        "preferred_model_type": preferred_model_type,
    }


def _pipeline_from_transition(
    first_capability: str,
    second_capability: str,
    reason: str,
    second_purpose: str,
    supported: set[str],
) -> Optional[Dict[str, Any]]:
    if first_capability not in supported or second_capability not in supported:
        return None
    first_artifact = STRUCTURED_ARTIFACTS_BY_CAPABILITY.get(first_capability, "structured intermediate artifact")
    second_input = f"{first_artifact} plus the original user goal"
    return {
        "should_chain": True,
        "reason": reason,
        "stages": [
            _stage(
                first_capability,
                f"Produce {first_artifact} to reduce the visual search space.",
                "Full image or scene.",
                first_artifact,
                "fast_detector" if first_capability == "object_detection" else "specialized_extractor",
            ),
            _stage(
                second_capability,
                second_purpose,
                second_input,
                "Task-specific answer or action guidance.",
                "navigation_model" if second_capability == "navigation" else "reasoning_vlm",
            ),
        ],
        "artifact_value_score": 0.85,
        "information_reduction": "significant",
        "inference_source": "router_transition_heuristic",
    }


def _routing_task_names_by_weight(routing_analysis: Optional[Dict[str, Any]]) -> List[str]:
    if not routing_analysis:
        return []
    weighted: Dict[str, float] = {}
    for task in routing_analysis.get("tasks", []):
        name = str(task.get("name", "")).strip().lower()
        if not name:
            continue
        weighted[name] = weighted.get(name, 0.0) + float(task.get("weight", 0.0) or 0.0)
    return [name for name, _ in sorted(weighted.items(), key=lambda item: item[1], reverse=True)]


def _default_pipeline_analysis(reason: str) -> Dict[str, Any]:
    return _with_pipeline_score({
        "should_chain": False,
        "reason": reason,
        "stages": [],
        "artifact_value_score": 0.0,
        "information_reduction": "none",
    })


def _build_two_stage_pipeline(
    producer: str,
    consumer: str,
    reason: str,
    consumer_purpose: str,
) -> Dict[str, Any]:
    producer_artifact = STRUCTURED_ARTIFACTS_BY_CAPABILITY.get(producer, "structured intermediate artifact")
    consumer_input = f"{producer_artifact} plus the original user goal"
    return {
        "should_chain": True,
        "reason": reason,
        "stages": [
            _stage(
                producer,
                f"Produce {producer_artifact} for downstream reasoning or guidance.",
                "Full image or scene.",
                producer_artifact,
                "fast_detector" if producer == "object_detection" else "specialized_extractor",
            ),
            _stage(
                consumer,
                consumer_purpose,
                consumer_input,
                "Task-specific answer or guidance.",
                "navigation_model" if consumer == "navigation" else "reasoning_vlm",
            ),
        ],
        "artifact_value_score": 0.85,
        "information_reduction": "significant",
        "inference_source": "artifact_flow_rewrite",
    }


def _choose_pipeline_by_score(original: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not candidate.get("should_chain"):
        return original
    if not original.get("should_chain"):
        return candidate

    original_score = float(original.get("pipeline_score", 0.0) or 0.0)
    candidate_score = float(candidate.get("pipeline_score", 0.0) or 0.0)
    original_useful = int(original.get("useful_transition_count", 0) or 0)
    candidate_useful = int(candidate.get("useful_transition_count", 0) or 0)

    if candidate_score > original_score + PIPELINE_REWRITE_MARGIN:
        return candidate
    if original_useful == 0 and candidate_useful > 0 and candidate_score >= MIN_PIPELINE_SCORE:
        return candidate
    return original


def infer_pipeline_from_information_reduction(
    task_text: str,
    routing_analysis: Optional[Dict[str, Any]],
    pipeline_analysis: Dict[str, Any],
    supported_capabilities: Iterable[str],
) -> Dict[str, Any]:
    if not routing_analysis:
        return pipeline_analysis

    supported = {str(capability).strip().lower() for capability in supported_capabilities if str(capability).strip()}
    task_names = set(_routing_task_names_by_weight(routing_analysis))
    text = task_text.lower()

    if not task_names:
        return _default_pipeline_analysis("No valid routing tasks were provided for artifact-driven pipeline synthesis.")

    # Merge overlap: camera framing guidance usually consumes detector output directly.
    if "camera_motion" in task_names and "spatial_relationship" in task_names:
        task_names.discard("spatial_relationship")

    # Distinguish camera framing from physical wayfinding.
    if "camera_motion" in task_names and "navigation" in task_names and _contains_any(text, CAMERA_GUIDANCE_KEYWORDS):
        if not _contains_any(text, WAYFINDING_KEYWORDS):
            task_names.discard("navigation")

    if _is_holistic_without_target_reduction(task_text) and "ocr" not in task_names:
        return _default_pipeline_analysis(
            "Holistic task without clear intermediate artifact reduction; prefer single-model execution.",
        )
    if "chart" in text and not _contains_any(text, TARGET_REDUCTION_KEYWORDS):
        return _default_pipeline_analysis(
            "Chart interpretation is holistic unless a structured artifact reduces downstream work.",
        )
    if (
        "ocr" in task_names
        and not _contains_any(text, TEXT_ANALYSIS_KEYWORDS)
        and task_names.issubset({"ocr", "general_reasoning"})
    ):
        return _default_pipeline_analysis(
            "Simple OCR does not need an additional reasoning stage without an explicit analysis or summary request.",
        )

    def _supports(capability: str) -> bool:
        return capability in supported

    # Highest-priority transition: detect target, then center/aim camera.
    if "camera_motion" in task_names and _supports("camera_motion") and _supports("object_detection"):
        if _contains_any(text, CAMERA_GUIDANCE_KEYWORDS) and (
            "object_detection" in task_names or _contains_any(text, TARGET_REDUCTION_KEYWORDS)
        ):
            return _choose_pipeline_by_score(
                pipeline_analysis,
                normalize_pipeline_analysis(_build_two_stage_pipeline(
                    "object_detection",
                    "camera_motion",
                    "Object detection first provides target coordinates that camera guidance consumes directly.",
                    "Use target coordinates to issue camera centering/framing guidance.",
                ),
                supported,
                ),
            )

    # Bus entrance requests benefit from preserving both the bus bbox and entrance waypoint artifacts.
    if (
        "object_detection" in task_names
        and _contains_any(text, ("bus",))
        and _contains_any(text, ("entrance",))
        and _supports("spatial_relationship")
        and _supports("navigation")
    ):
        return _choose_pipeline_by_score(
            pipeline_analysis,
            normalize_pipeline_analysis({
                "should_chain": True,
                "reason": (
                    "Detect the correct bus, localize the entrance relative to that bus, "
                    "then provide guidance using the entrance waypoint."
                ),
                "stages": [
                    _stage(
                        "object_detection",
                        "Identify the correct bus and produce its bounding box.",
                        "Full image or scene.",
                        "bus bounding box and coordinates",
                        "fast_detector",
                    ),
                    _stage(
                        "spatial_relationship",
                        "Locate the entrance relative to the detected bus.",
                        "Bus bounding box and current scene.",
                        "entrance waypoint relative to bus",
                        "reasoning_vlm",
                    ),
                    _stage(
                        "navigation",
                        "Use the entrance waypoint to guide or orient the user.",
                        "Entrance waypoint plus current scene.",
                        "movement or orientation guidance",
                        "navigation_model",
                    ),
                ],
                "artifact_value_score": 1.0,
                "information_reduction": "significant",
                "inference_source": "artifact_flow_rewrite",
            }, supported),
        )

    # Find-then-guide tasks: detect target before navigation.
    if "navigation" in task_names and _supports("navigation") and _supports("object_detection"):
        if "object_detection" in task_names or _contains_any(text, TARGET_REDUCTION_KEYWORDS):
            if _contains_any(text, ("bus", "entrance")) and _supports("spatial_relationship"):
                return _choose_pipeline_by_score(
                    pipeline_analysis,
                    normalize_pipeline_analysis({
                        "should_chain": True,
                        "reason": (
                            "Detect the target object, localize the relevant entrance or access point, "
                            "then guide the user using that narrowed waypoint."
                        ),
                        "stages": [
                            _stage(
                                "object_detection",
                                "Detect the relevant bus or destination object.",
                                "Full image or scene.",
                                "target bounding box and object coordinates",
                                "fast_detector",
                            ),
                            _stage(
                                "spatial_relationship",
                                "Localize the entrance relative to the detected object.",
                                "Detected target bounding box plus the scene.",
                                "entrance location or waypoint",
                                "reasoning_vlm",
                            ),
                            _stage(
                                "navigation",
                                "Guide the user toward the localized entrance waypoint.",
                                "Entrance waypoint and current scene.",
                                "Movement guidance.",
                                "navigation_model",
                            ),
                        ],
                        "artifact_value_score": 0.9,
                        "information_reduction": "significant",
                        "inference_source": "artifact_flow_rewrite",
                    }, supported),
                )
            return _choose_pipeline_by_score(
                pipeline_analysis,
                normalize_pipeline_analysis(_build_two_stage_pipeline(
                    "object_detection",
                    "navigation",
                    "Navigation consumes detected target location to avoid full-scene search.",
                    "Use detected target location to produce movement or wayfinding guidance.",
                ),
                supported,
                ),
            )

    # Detect target then perform structured layout/symbol interpretation on that narrowed region.
    if "object_detection" in task_names and "map_web" in task_names and _supports("map_web") and _supports("object_detection"):
        if _contains_any(text, TARGET_REDUCTION_KEYWORDS):
            return _choose_pipeline_by_score(
                pipeline_analysis,
                normalize_pipeline_analysis(_build_two_stage_pipeline(
                    "object_detection",
                    "map_web",
                    "Object detection narrows the visual region before map/layout interpretation.",
                    "Interpret structured visual layout relative to the detected target region.",
                ),
                supported,
                ),
            )

    # Read-then-understand tasks.
    if "ocr" in task_names and _supports("ocr"):
        if "general_reasoning" in task_names and _supports("general_reasoning") and _contains_any(text, TEXT_ANALYSIS_KEYWORDS):
            return _choose_pipeline_by_score(
                pipeline_analysis,
                normalize_pipeline_analysis(_build_two_stage_pipeline(
                    "ocr",
                    "general_reasoning",
                    "OCR extracts text that reasoning consumes for summary/interpretation.",
                    "Summarize or interpret the extracted text for the user.",
                ),
                supported,
                ),
            )
        if "map_web" in task_names and _supports("map_web") and _contains_any(text, TEXT_ANALYSIS_KEYWORDS):
            return _choose_pipeline_by_score(
                pipeline_analysis,
                normalize_pipeline_analysis(_build_two_stage_pipeline(
                    "ocr",
                    "map_web",
                    "OCR extracts labels/text that map/layout interpretation can consume.",
                    "Interpret structured visual content using extracted text anchors.",
                ),
                supported,
                ),
            )

    # Detector output can directly feed reasoning if the goal is target-focused interpretation.
    if "object_detection" in task_names and "general_reasoning" in task_names and _supports("general_reasoning") and _supports("object_detection"):
        if _contains_any(text, TARGET_REDUCTION_KEYWORDS):
            return _choose_pipeline_by_score(
                pipeline_analysis,
                normalize_pipeline_analysis(_build_two_stage_pipeline(
                    "object_detection",
                    "general_reasoning",
                    "Object detection narrows the visual scope for downstream reasoning.",
                    "Reason over the detected target region to produce final guidance.",
                ),
                supported,
                ),
            )

    # Keep pipeline small and artifact-driven; otherwise prefer single model.
    if pipeline_analysis.get("should_chain"):
        return pipeline_analysis
    return _default_pipeline_analysis(
        "No clear producer-consumer artifact dependency was found; prefer a minimal single-model plan.",
    )


def _dynamic_score_weights(routing_analysis: Dict[str, Any]) -> Dict[str, float]:
    task_component = 0.85
    latency_component = 0.15

    latency = routing_analysis.get("latency_sensitivity", {})

    if latency.get("level") == "high":
        boost = 0.25 * float(latency.get("weight", 0.5))
        latency_component += boost
        task_component -= boost

    if latency.get("level") == "low":
        reduction = 0.10 * float(latency.get("weight", 0.5))
        latency_component -= reduction
        task_component += reduction

    task_component = max(task_component, 0.60)
    latency_component = max(latency_component, 0.05)
    total = task_component + latency_component
    return {
        "task": task_component / total,
        "latency": latency_component / total,
    }


def _routing_task_match_score(model: ModelProfile, routing_analysis: Dict[str, Any]) -> float:
    score = 0.0
    for task in routing_analysis.get("tasks", []):
        score += float(task.get("weight", 0.0)) * _task_strength(model, task.get("name", ""))
    return float(max(0.0, min(1.0, score)))


def _score_with_routing_analysis(model: ModelProfile, routing_analysis: Dict[str, Any]) -> Dict[str, float]:
    mix = _dynamic_score_weights(routing_analysis)
    task_match = _routing_task_match_score(model, routing_analysis)
    latency_match = float(max(0.0, min(1.0, model.latency)))

    final_score = (
        task_match * mix["task"]
        + latency_match * mix["latency"]
    )
    return {
        "task_match_score": task_match,
        "latency_match_score": latency_match,
        "task_component_weight": mix["task"],
        "latency_component_weight": mix["latency"],
        "final_score": final_score,
    }


def compute_capability_weights(
    task_text: str,
    capability_descriptions: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    descriptions = capability_descriptions or load_capability_descriptions()
    if not descriptions:
        return {}
    if DEFAULT_FALLBACK_CAPABILITY in descriptions:
        return {DEFAULT_FALLBACK_CAPABILITY: 1.0}
    # Deterministic fallback when general_reasoning is absent.
    first_capability = next(iter(descriptions))
    return {str(first_capability): 1.0}


def score_model(model: ModelProfile, capability_weights: Dict[str, float]) -> float:
    return sum(
        weight * float(model.capabilities.get(capability, 0.0))
        for capability, weight in capability_weights.items()
    )


def latency_penalty(latency_ms: float) -> float:
    return 1.0 + max(float(latency_ms), 0.0) / 5000.0


def _rank_profile_rows(
    profiles: List[ModelProfile],
    capability_weights: Dict[str, float],
    routing_analysis: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ranking = []
    for profile in profiles:
        if routing_analysis:
            score = _score_with_routing_analysis(profile, routing_analysis)
            row = {
                "model": profile.name,
                "type": profile.type,
                "source": profile.source,
                "latency_ms": profile.latency_ms,
                "latency": profile.latency,
                **score,
            }
        else:
            capability_score = score_model(profile, capability_weights)
            penalty = latency_penalty(profile.latency_ms)
            row = {
                "model": profile.name,
                "type": profile.type,
                "source": profile.source,
                "latency_ms": profile.latency_ms,
                "capability_score": capability_score,
                "latency_penalty": penalty,
                "final_score": capability_score / penalty,
            }
        ranking.append(row)

    ranking.sort(
        key=lambda item: (
            item["final_score"],
            item.get("capability_score", item.get("task_match_score", 0.0)),
            -item["latency_ms"],
            item["model"],
        ),
        reverse=True,
    )
    return ranking


def _stage_routing_analysis(capability: str, latency_sensitivity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "tasks": [{"name": capability, "weight": 1.0, "reason": "Pipeline stage capability."}],
        "latency_sensitivity": latency_sensitivity or {"level": "medium", "weight": 0.5, "reason": ""},
    }


def _build_pipeline_execution_plan(
    task_text: str,
    profiles: List[ModelProfile],
    pipeline_analysis: Dict[str, Any],
    routing_analysis: Optional[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stage_selections = []
    plan_stages = []
    latency_sensitivity = (routing_analysis or {}).get("latency_sensitivity")

    for index, stage in enumerate(pipeline_analysis.get("stages", []), start=1):
        capability = str(stage.get("capability", "")).strip().lower()
        stage_routing = _stage_routing_analysis(capability, latency_sensitivity)
        ranking = _rank_profile_rows(profiles, {capability: 1.0}, stage_routing)
        selected = ranking[0] if ranking else None
        selection = {
            "index": index,
            "capability": capability,
            "selected_model": selected.get("model") if selected else None,
            "selected": selected,
            "ranking": ranking,
        }
        stage_selections.append(selection)
        plan_stages.append({
            "index": index,
            **stage,
            "selected_model": selection["selected_model"],
            "selected": selected,
        })

    return stage_selections, {
        "mode": "pipeline",
        "task": task_text,
        "reason": pipeline_analysis.get("reason", ""),
        "stages": plan_stages,
    }


def rank_models(
    task_text: str,
    models: Optional[Iterable[ModelProfile]] = None,
    capability_descriptions: Optional[Dict[str, List[str]]] = None,
    routing_analysis: Optional[Dict[str, Any]] = None,
    pipeline_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    descriptions = capability_descriptions or load_capability_descriptions()
    profiles = list(models) if models is not None else load_model_profiles()
    weights = compute_capability_weights(task_text, descriptions)
    normalized_routing = normalize_routing_analysis(routing_analysis, descriptions.keys())
    normalized_pipeline = normalize_pipeline_analysis(pipeline_analysis, descriptions.keys())
    normalized_pipeline = infer_pipeline_from_information_reduction(
        task_text,
        normalized_routing,
        normalized_pipeline,
        descriptions.keys(),
    )
    ranking = []

    logger.info("[Model Router] request=%s", task_text)
    if normalized_routing:
        task_log = [
            {"name": task["name"], "weight": round(task["weight"], 3)}
            for task in normalized_routing["tasks"]
        ]
        logger.info("[Model Router] capability_distribution=%s", json.dumps(task_log, sort_keys=True))
        logger.info(
            "[Model Router] latency_sensitivity=%s",
            json.dumps(normalized_routing.get("latency_sensitivity", {}), sort_keys=True),
        )
    else:
        fallback_distribution = {
            key: round(value, 3)
            for key, value in sorted(weights.items(), key=lambda item: item[1], reverse=True)
            if value > 0.001
        }
        logger.info("[Model Router] capability_distribution=%s", json.dumps(fallback_distribution, sort_keys=True))
        logger.info(
            "[Model Router] latency_sensitivity=%s",
            json.dumps({"level": "medium", "weight": 0.5, "reason": "fallback routing without routing_analysis"}, sort_keys=True),
        )

    logger.info(
        "[Model Router] pipeline_decision=%s",
        json.dumps({
            "should_chain": normalized_pipeline["should_chain"],
            "reason": normalized_pipeline["reason"],
            "artifact_value_score": normalized_pipeline.get("artifact_value_score", 0.0),
            "information_reduction": normalized_pipeline.get("information_reduction", "none"),
            "stage_count": normalized_pipeline.get("stage_count", 0),
            "artifact_gain": normalized_pipeline.get("artifact_gain", 0.0),
            "complexity_penalty": normalized_pipeline.get("complexity_penalty", 0.0),
            "pipeline_score": normalized_pipeline.get("pipeline_score", 0.0),
        }, sort_keys=True),
    )
    logger.info("[Model Router] pipeline_stages=%s", json.dumps(normalized_pipeline["stages"], sort_keys=True))

    ranking = _rank_profile_rows(profiles, weights, normalized_routing)
    for row in ranking:
        logger.info(
            "[Model Router] model_score model=%s task_match=%.4f latency_match=%.4f final_score=%.4f",
            row["model"],
            row.get("capability_score", row.get("task_match_score", 0.0)),
            row.get("latency_match_score", 1.0 / row.get("latency_penalty", 1.0)),
            row["final_score"],
        )

    selected = ranking[0] if ranking else None
    stage_model_selection = []
    if normalized_pipeline["should_chain"]:
        stage_model_selection, final_execution_plan = _build_pipeline_execution_plan(
            task_text,
            profiles,
            normalized_pipeline,
            normalized_routing,
        )
        selected = None
        for stage_selection in stage_model_selection:
            logger.info(
                "[Model Router] stage_model_selection=%s",
                json.dumps({
                    "index": stage_selection["index"],
                    "capability": stage_selection["capability"],
                    "selected_model": stage_selection["selected_model"],
                    "final_score": (
                        round(float(stage_selection["selected"].get("final_score", 0.0)), 4)
                        if stage_selection.get("selected")
                        else None
                    ),
                }, sort_keys=True),
            )
    else:
        final_execution_plan = {
            "mode": "single_model",
            "task": task_text,
            "selected_model": selected["model"] if selected else None,
            "selected": selected,
        }

    logger.info(
        "[Model Router] final_execution_plan=%s",
        json.dumps({
            "mode": final_execution_plan["mode"],
            "selected_model": final_execution_plan.get("selected_model"),
            "stages": [
                {
                    "index": stage.get("index"),
                    "capability": stage.get("capability"),
                    "selected_model": stage.get("selected_model"),
                }
                for stage in final_execution_plan.get("stages", [])
            ],
        }, sort_keys=True),
    )

    if selected:
        logger.info(
            "[Model Router] selected_model=%s final_score=%.4f",
            selected["model"],
            selected["final_score"],
        )
    return {
        "task": task_text,
        "capability_weights": weights,
        "routing_analysis": normalized_routing,
        "pipeline_analysis": normalized_pipeline,
        "ranking": ranking,
        "selected_model": selected["model"] if selected else None,
        "selected": selected,
        "stage_model_selection": stage_model_selection,
        "final_execution_plan": final_execution_plan,
    }


def select_model(
    task_text: str,
    routing_analysis: Optional[Dict[str, Any]] = None,
    pipeline_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return rank_models(task_text, routing_analysis=routing_analysis, pipeline_analysis=pipeline_analysis)


def _text_parts_from_content(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return parts
    return []


def _task_text(
    capability: Optional[str],
    task: Optional[str],
    task_category: Optional[str],
    messages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> str:
    if isinstance(metadata, dict):
        route_text = metadata.get("route_text") or metadata.get("routing_text")
        if isinstance(route_text, str) and route_text.strip():
            return route_text.strip()

    parts = [value for value in (task, task_category, capability) if isinstance(value, str) and value.strip()]
    for message in messages:
        parts.extend(_text_parts_from_content(message.get("content")))
    return " ".join(parts).strip()[:6000]


def system_llm_call(
    messages: Optional[List[Dict[str, Any]]] = None,
    images: Optional[Iterable[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **_: Any,
):
    """Call the fixed infrastructure model without semantic routing."""
    logger.info("[System LLM] model=%s", SYSTEM_MODEL)
    return call_model(SYSTEM_MODEL, messages or [], images=images, metadata=metadata)


def copilot_llm_call(
    capability: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    images: Optional[Iterable[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
    task_category: Optional[str] = None,
):
    """Route a Copilot/tool-generation request, then call the selected model."""
    messages = messages or []
    route_text = _task_text(capability, task, task_category, messages, metadata)
    callable_profiles = [profile for profile in load_model_profiles() if profile.model]
    routing_analysis = metadata.get("routing_analysis") if isinstance(metadata, dict) else None
    route = rank_models(route_text, callable_profiles, routing_analysis=routing_analysis)
    selected_profile = next(profile for profile in callable_profiles if profile.name == route["selected_model"])
    weights = {key: round(value, 3) for key, value in route["capability_weights"].items() if value > 0.005}

    logger.info(
        "[Copilot Router] task=%s selected_model=%s capability_weights=%s routing_analysis=%s",
        route_text,
        route["selected_model"],
        json.dumps(weights, sort_keys=True),
        json.dumps(route.get("routing_analysis"), sort_keys=True) if route.get("routing_analysis") else "null",
    )
    return call_model(selected_profile.model, messages, images=images, metadata=metadata)


__all__ = [
    "BACKEND_DIR",
    "MODEL_PROFILES_PATH",
    "CAPABILITY_PROFILES_PATH",
    "SYSTEM_MODEL",
    "ModelProfile",
    "load_capability_descriptions",
    "load_model_profiles",
    "compute_capability_weights",
    "normalize_routing_analysis",
    "normalize_pipeline_analysis",
    "infer_pipeline_from_information_reduction",
    "score_model",
    "latency_penalty",
    "rank_models",
    "select_model",
    "system_llm_call",
    "copilot_llm_call",
]
