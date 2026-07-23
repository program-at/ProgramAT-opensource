"""Resolve tool policies and connect the policy executor to provider adapters."""

from __future__ import annotations

import logging
import os
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
    on_progress=None, on_progress_error=None, is_cancelled=None,
) -> str:
    resolved = resolve_tool_policy(policy)
    progressive = resolved.get("strategy") == "parallel_progressive"

    def invoke(model_name: str, task: str, call_image: Any, call_metadata: Mapping[str, Any]) -> str:
        profile = _profile(model_name)
        adapter = IMPLEMENTATION_EXECUTORS[profile.kind]
        timeout_seconds = call_metadata.get("timeout")
        cancelled = call_metadata.get("_progressive_cancelled")
        if progressive:
            logger.info(
                "[Progressive] model_call_start invocation_id=%s model=%s adapter=%s "
                "provider_model=%s timeout_seconds=%.1f cancelled=%s",
                call_metadata.get("request_id", "unknown"), model_name, profile.kind,
                profile.model or profile.model_name, float(timeout_seconds),
                bool(callable(cancelled) and cancelled()),
            )
        try:
            if callable(cancelled) and cancelled():
                raise StreamingExecutionCancelled(f"{model_name} cancelled before provider call")
            result = adapter(
                profile,
                [{"role": "user", "content": task}],
                [] if call_image is None else [call_image],
                dict(call_metadata),
            )
            if callable(cancelled) and cancelled():
                raise StreamingExecutionCancelled(f"{model_name} cancelled after provider call")
        except Exception:
            if progressive:
                logger.exception(
                    "[Progressive] model_call_failure invocation_id=%s model=%s adapter=%s "
                    "provider_model=%s timeout_seconds=%.1f cancelled=%s",
                    call_metadata.get("request_id", "unknown"), model_name, profile.kind,
                    profile.model or profile.model_name, float(timeout_seconds),
                    bool(callable(cancelled) and cancelled()),
                )
            raise
        answer = _response_text(result.response)
        if not answer:
            raise RuntimeError(f"Model {model_name!r} returned an empty response")
        if progressive:
            logger.info(
                "[Progressive] model_call_complete invocation_id=%s model=%s adapter=%s "
                "provider_model=%s timeout_seconds=%.1f cancelled=false",
                call_metadata.get("request_id", "unknown"), model_name, profile.kind,
                profile.model or profile.model_name, float(timeout_seconds),
            )
        return answer

    call_metadata = dict(metadata or {})
    call_metadata.update({"request_id": request_id or uuid.uuid4().hex, "num_retries": 0})
    if progressive:
        progressive_timeout = max(
            0.01, float(os.environ.get("PROGRESSIVE_MODEL_TIMEOUT_SECONDS", "60"))
        )
        call_metadata.update({
            "timeout": progressive_timeout,
            "progressive_model_timeout_seconds": progressive_timeout,
            "_progressive_cancelled": is_cancelled,
        })
    return execute_tool_policy(
        resolved, prompt, image, invoke, metadata=call_metadata,
        on_progress=on_progress, on_progress_error=on_progress_error,
        is_cancelled=is_cancelled,
    )


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
