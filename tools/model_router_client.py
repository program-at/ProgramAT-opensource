"""Tool-facing client for ProgramAT's centralized backend model router.

Generated tools should import from this module instead of importing provider
SDKs, detector libraries, or router internals directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_BACKEND_ROUTER_PATH = Path(__file__).resolve().parent.parent / "backend" / "model_router.py"
_BACKEND_DIR = _BACKEND_ROUTER_PATH.parent

_backend_model_router = sys.modules.get("model_router")

if _backend_model_router is None:
    if str(_BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(_BACKEND_DIR))

    _SPEC = importlib.util.spec_from_file_location("_programat_backend_model_router", _BACKEND_ROUTER_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Could not load backend model router from {_BACKEND_ROUTER_PATH}")

    _backend_model_router = importlib.util.module_from_spec(_SPEC)
    sys.modules.setdefault("_programat_backend_model_router", _backend_model_router)
    _SPEC.loader.exec_module(_backend_model_router)

llm_call = _backend_model_router.llm_call
vision_call = _backend_model_router.vision_call
detect_objects = _backend_model_router.detect_objects
ocr_call = _backend_model_router.ocr_call


def routed_llm_call(
    task: str,
    messages: List[Dict[str, Any]],
    images: Optional[Iterable[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Route a text or multimodal LLM/VLM request through the backend router."""
    return llm_call(task=task, messages=messages, images=images, metadata=metadata)


def routed_vision_call(
    task: str,
    image: Any,
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    system_prompt: str = "Keep responses concise and audio-friendly.",
):
    """Route a single-image vision-language request through the backend router."""
    return vision_call(
        task=task,
        image=image,
        prompt=prompt,
        metadata=metadata,
        system_prompt=system_prompt,
    )


def routed_object_detection(
    task: str,
    image: Any,
    labels: Optional[Iterable[str]] = None,
    confidence: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Route detection/localization work through the backend router."""
    return detect_objects(
        task=task,
        image=image,
        labels=labels,
        confidence=confidence,
        metadata=metadata,
    )


def routed_ocr_call(
    image: Any,
    language_hints: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Route OCR/text extraction through the backend router."""
    return ocr_call(image=image, language_hints=language_hints, metadata=metadata)


__all__ = [
    "llm_call",
    "vision_call",
    "detect_objects",
    "ocr_call",
    "routed_llm_call",
    "routed_vision_call",
    "routed_object_detection",
    "routed_ocr_call",
]
