"""Semantic capability router and simple LLM call entrypoints for ProgramAT."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from litellm_utils import call_model


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PROFILES_PATH = BACKEND_DIR / "model_profiles.yaml"
CAPABILITY_PROFILES_PATH = BACKEND_DIR / "capability_profiles.yaml"
SYSTEM_MODEL = os.environ.get("SYSTEM_LLM_MODEL", "gemini/gemini-2.0-flash-preview")
CAPABILITY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CAPABILITY_SIMILARITIES = 2


@dataclass(frozen=True)
class ModelProfile:
    name: str
    type: str
    latency_ms: float
    source: str
    capabilities: Dict[str, float]
    model: str = ""


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_capability_descriptions(path: Path = CAPABILITY_PROFILES_PATH) -> Dict[str, List[str]]:
    data = _load_yaml(path)
    capabilities = data.get("capabilities", {})
    if not isinstance(capabilities, dict) or not capabilities:
        raise ValueError(f"No capabilities configured in {path}")

    descriptions: Dict[str, List[str]] = {}
    for capability, raw in capabilities.items():
        capability_name = str(capability)
        if isinstance(raw, list):
            values = [str(value).strip() for value in raw if str(value).strip()]
            if not values:
                raise ValueError(f"Capability {capability_name!r} must contain a non-empty list of descriptions")
            descriptions[capability_name] = values
            continue

        if not isinstance(raw, dict):
            raise ValueError(
                f"Capability {capability_name!r} must be a list or mapping with description/include_examples/exclude_examples"
            )

        description = str(raw.get("description", "")).strip()
        include_examples = raw.get("include_examples", [])
        exclude_examples = raw.get("exclude_examples", [])
        if include_examples is None:
            include_examples = []
        if exclude_examples is None:
            exclude_examples = []
        if not isinstance(include_examples, list) or not isinstance(exclude_examples, list):
            raise ValueError(
                f"Capability {capability_name!r} include_examples/exclude_examples must be lists"
            )

        values = []
        if description:
            values.append(description)
        values.extend(str(value).strip() for value in include_examples if str(value).strip())
        if not values:
            raise ValueError(
                f"Capability {capability_name!r} must provide description and/or include_examples"
            )
        descriptions[capability_name] = values
    return descriptions


def load_model_profiles(path: Path = MODEL_PROFILES_PATH) -> List[ModelProfile]:
    data = _load_yaml(path)
    models = data.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No models configured in {path}")

    profiles: List[ModelProfile] = []
    for name, raw_profile in models.items():
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Model {name!r} must be a mapping")
        capabilities = raw_profile.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise ValueError(f"Model {name!r} capabilities must be a mapping")
        profiles.append(ModelProfile(
            name=str(name),
            type=str(raw_profile.get("type", "general_vlm")),
            latency_ms=float(raw_profile.get("latency_ms", 1000)),
            source=str(raw_profile.get("source", "unknown")),
            capabilities={str(key): float(value or 0.0) for key, value in capabilities.items()},
            model=str(raw_profile.get("model", "") or ""),
        ))
    return profiles


def _normalized_capability_descriptions(
    capability_descriptions: Dict[str, List[str]],
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    return tuple(
        (capability, tuple(values))
        for capability, values in capability_descriptions.items()
    )


def _average_top_k(values: List[float], k: int = TOP_K_CAPABILITY_SIMILARITIES) -> float:
    if not values:
        return 0.0
    top_values = sorted(values, reverse=True)[: max(1, k)]
    return float(sum(top_values) / len(top_values))


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(CAPABILITY_EMBEDDING_MODEL)


@lru_cache(maxsize=32)
def _precomputed_capability_embeddings(
    normalized_descriptions: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> Dict[str, np.ndarray]:
    flattened: List[Tuple[str, str]] = [
        (capability, description)
        for capability, values in normalized_descriptions
        for description in values
    ]
    if not flattened:
        return {capability: np.empty((0, 0), dtype=np.float32) for capability, _ in normalized_descriptions}

    model = _embedding_model()
    description_embeddings = model.encode(
        [description for _, description in flattened],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    grouped: Dict[str, List[np.ndarray]] = {capability: [] for capability, _ in normalized_descriptions}
    for (capability, _), embedding in zip(flattened, description_embeddings):
        grouped[capability].append(embedding)

    return {
        capability: np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)
        for capability, embeddings in grouped.items()
    }


def _capability_similarities(
    task_text: str,
    capability_descriptions: Dict[str, List[str]],
) -> Dict[str, List[float]]:
    normalized_descriptions = _normalized_capability_descriptions(capability_descriptions)
    if not task_text.strip() or not normalized_descriptions:
        return {capability: [] for capability in capability_descriptions}

    model = _embedding_model()
    task_embedding = model.encode(
        [task_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    capability_embeddings = _precomputed_capability_embeddings(normalized_descriptions)

    similarities: Dict[str, List[float]] = {capability: [] for capability in capability_descriptions}
    for capability, description_vectors in capability_embeddings.items():
        if description_vectors.size == 0:
            continue
        scores = description_vectors @ task_embedding
        similarities[capability] = [float(score) for score in scores.tolist()]
    return similarities


def compute_capability_weights(
    task_text: str,
    capability_descriptions: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    descriptions = capability_descriptions or load_capability_descriptions()
    similarities = _capability_similarities(task_text, descriptions)
    raw_weights = {
        capability: max(0.0, _average_top_k(values))
        for capability, values in similarities.items()
    }
    # Keep only the top 3 capabilities; zero out the rest before normalizing.
    top3 = set(
        capability for capability, _ in
        sorted(raw_weights.items(), key=lambda item: item[1], reverse=True)[:3]
    )
    raw_weights = {
        capability: (weight if capability in top3 else 0.0)
        for capability, weight in raw_weights.items()
    }
    total = sum(raw_weights.values())
    if total <= 0.0:
        capability_count = len(descriptions)
        if capability_count == 0:
            return {}
        uniform_weight = 1.0 / capability_count
        return {capability: uniform_weight for capability in descriptions}
    return {capability: raw_weights[capability] / total for capability in descriptions}


def score_model(model: ModelProfile, capability_weights: Dict[str, float]) -> float:
    return sum(
        weight * float(model.capabilities.get(capability, 0.0))
        for capability, weight in capability_weights.items()
    )


def latency_penalty(latency_ms: float) -> float:
    return 1.0 + max(float(latency_ms), 0.0) / 5000.0


def rank_models(
    task_text: str,
    models: Optional[Iterable[ModelProfile]] = None,
    capability_descriptions: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    descriptions = capability_descriptions or load_capability_descriptions()
    profiles = list(models) if models is not None else load_model_profiles()
    weights = compute_capability_weights(task_text, descriptions)
    ranking = []

    for profile in profiles:
        capability_score = score_model(profile, weights)
        penalty = latency_penalty(profile.latency_ms)
        final_score = capability_score / penalty
        ranking.append({
            "model": profile.name,
            "type": profile.type,
            "source": profile.source,
            "latency_ms": profile.latency_ms,
            "capability_score": capability_score,
            "latency_penalty": penalty,
            "final_score": final_score,
        })

    ranking.sort(
        key=lambda item: (
            item["final_score"],
            item["capability_score"],
            -item["latency_ms"],
            item["model"],
        ),
        reverse=True,
    )

    selected = ranking[0] if ranking else None
    return {
        "task": task_text,
        "capability_weights": weights,
        "ranking": ranking,
        "selected_model": selected["model"] if selected else None,
        "selected": selected,
    }


def select_model(task_text: str) -> Dict[str, Any]:
    return rank_models(task_text)


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
    route = rank_models(route_text, callable_profiles)
    selected_profile = next(profile for profile in callable_profiles if profile.name == route["selected_model"])
    weights = {key: round(value, 3) for key, value in route["capability_weights"].items() if value > 0.005}

    logger.info(
        "[Copilot Router] task=%s selected_model=%s capability_weights=%s",
        route_text,
        route["selected_model"],
        json.dumps(weights, sort_keys=True),
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
    "score_model",
    "latency_penalty",
    "rank_models",
    "select_model",
    "system_llm_call",
    "copilot_llm_call",
]
