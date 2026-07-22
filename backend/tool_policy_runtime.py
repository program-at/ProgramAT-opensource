"""Resolve tool policies and connect the policy executor to provider adapters."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any, Iterable, Mapping, Optional

from litellm_utils import call_model
from model_adapters import (
    IMPLEMENTATION_EXECUTORS,
    ImplementationProfile,
    StreamingExecutionCancelled,
    _response_text,
)
from model_registry import DEFAULT_TAKE_PHOTO_POLICY, MODEL_REGISTRY, SYSTEM_MODEL
from policy_executor import ToolPolicyError, execute_tool_policy, validate_tool_policy

logger = logging.getLogger(__name__)
STREAMING_EXECUTION_CONTEXT: ContextVar[bool] = ContextVar("streaming_execution", default=False)
TOOL_EXECUTION_IMAGES: ContextVar[Optional[list[Any]]] = ContextVar("tool_execution_images", default=None)


def resolve_tool_policy(policy: Any = None) -> dict[str, Any]:
    """Resolve an explicit policy, safely falling back to the default when invalid."""
    if policy is not None:
        try:
            return validate_tool_policy(policy)
        except (ToolPolicyError, TypeError) as exc:
            logger.warning("[Tool Policy] invalid explicit policy; using default error=%s", exc)
    return dict(DEFAULT_TAKE_PHOTO_POLICY)


def _profile(model_name: str) -> ImplementationProfile:
    entry = MODEL_REGISTRY[model_name]
    return ImplementationProfile(
        name=model_name,
        kind=entry["executor"],
        model=entry["provider_model"] if entry["executor"] != "yolo" else "",
        model_name=entry["provider_model"] if entry["executor"] == "yolo" else "",
    )


def execute_resolved_tool_policy(
    prompt: str, image: Any, *, policy: Any = None,
    request_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    resolved = resolve_tool_policy(policy)

    def invoke(model_name: str, task: str, call_image: Any, call_metadata: Mapping[str, Any]) -> str:
        profile = _profile(model_name)
        adapter = IMPLEMENTATION_EXECUTORS[profile.kind]
        result = adapter(
            profile,
            [{"role": "user", "content": task}],
            [] if call_image is None else [call_image],
            dict(call_metadata),
        )
        answer = _response_text(result.response)
        if not answer:
            raise RuntimeError(f"Model {model_name!r} returned an empty response")
        return answer

    call_metadata = dict(metadata or {})
    call_metadata.update({"request_id": request_id or uuid.uuid4().hex, "num_retries": 0})
    return execute_tool_policy(resolved, prompt, image, invoke, metadata=call_metadata)


def system_llm_call(messages=None, images=None, metadata=None):
    logger.info("[System Model] model=%s", SYSTEM_MODEL)
    return call_model(SYSTEM_MODEL, messages or [], images=images, metadata=metadata)


def copilot_llm_call(capability=None, messages=None, images: Optional[Iterable[Any]] = None,
                     metadata=None, task=None, task_category=None, goal=None):
    del capability, task_category
    prompt_parts = [
        str(message.get("content") or "").strip() for message in (messages or [])
        if isinstance(message, dict) and message.get("role") == "user"
        and str(message.get("content") or "").strip()
    ]
    prompt_parts.extend(str(value).strip() for value in (task, goal) if str(value or "").strip())
    image_items = list(images or TOOL_EXECUTION_IMAGES.get() or [])
    if not prompt_parts or not image_items:
        raise ValueError("copilot_llm_call requires a task prompt and image")
    answer = execute_resolved_tool_policy(
        "\n\n".join(prompt_parts), image_items[0],
        policy=(metadata or {}).get("tool_policy"), metadata=metadata,
    )
    return {"response": answer, "artifact": {"text": answer}, "implementation": "tool_policy"}


__all__ = [
    "STREAMING_EXECUTION_CONTEXT", "StreamingExecutionCancelled", "TOOL_EXECUTION_IMAGES",
    "copilot_llm_call", "execute_resolved_tool_policy", "resolve_tool_policy", "system_llm_call",
]
