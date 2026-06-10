"""Tool-facing client for ProgramAT's centralized backend model router.

Generated tools should import from this module instead of importing provider
SDKs, detector libraries, or router internals directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from model_router import detect_objects, llm_call, ocr_call, vision_call
except ModuleNotFoundError:
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from model_router import detect_objects, llm_call, ocr_call, vision_call


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
    "routed_llm_call",
    "routed_vision_call",
    "routed_object_detection",
    "routed_ocr_call",
]
