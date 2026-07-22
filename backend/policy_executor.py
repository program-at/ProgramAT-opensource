"""Validation and shared execution of structured model tool policies."""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Callable, Mapping, Optional

from model_registry import MODEL_REGISTRY
from strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)
ModelCall = Callable[[str, str, Any, Mapping[str, Any]], str]


class ToolPolicyError(ValueError):
    pass


def validate_tool_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise ToolPolicyError("tool policy must be an object")
    resolved = dict(policy)
    strategy = str(resolved.get("strategy") or "").strip()
    spec = STRATEGY_REGISTRY.get(strategy)
    if not spec:
        raise ToolPolicyError(f"unknown execution strategy: {strategy!r}")
    missing = [field for field in spec["required"] if field not in resolved]
    if missing:
        raise ToolPolicyError(f"{strategy} policy is missing: {', '.join(missing)}")
    allowed = {"strategy", *spec["schema"]}
    unsupported = sorted(set(resolved) - allowed)
    if unsupported:
        raise ToolPolicyError(f"unsupported fields for {strategy}: {unsupported}")
    models = resolved.get("models", [])
    if strategy != "conditional" and (not isinstance(models, list) or not models):
        raise ToolPolicyError(f"{strategy} policy requires a non-empty models list")
    if strategy == "single" and len(models) != 1:
        raise ToolPolicyError("single policy requires exactly one model")
    if isinstance(models, list) and any(not isinstance(name, str) for name in models):
        raise ToolPolicyError("models must contain only model names")
    referenced = list(models) if isinstance(models, list) else []
    for field in ("evaluator", "aggregator"):
        if resolved.get(field):
            referenced.append(resolved[field])
            if not isinstance(resolved[field], str):
                raise ToolPolicyError(f"{field} must be a model name")
    unknown = [name for name in referenced if name not in MODEL_REGISTRY]
    if unknown:
        raise ToolPolicyError(f"unknown models: {unknown}")
    stop_condition = resolved.get("stop_condition")
    allowed_stops = {
        "cascade": {"accepted", "non_empty"},
        "parallel_first": {"accepted", "first_complete"},
    }
    if stop_condition is not None and stop_condition not in allowed_stops.get(strategy, set()):
        raise ToolPolicyError(f"unsupported stop_condition for {strategy}: {stop_condition!r}")
    if strategy == "conditional":
        validate_tool_policy(resolved["if_true"])
        validate_tool_policy(resolved["if_false"])
    return resolved


def _evaluation_prompt(task: str, answer: str) -> str:
    return (
        "Judge whether the candidate directly and usefully answers the task. "
        "Reply exactly YES or NO.\n\nTask:\n" + task + "\n\nCandidate:\n" + answer
    )


def execute_tool_policy(
    policy: Mapping[str, Any], prompt: str, image: Any, model_call: ModelCall,
    *, metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    policy = validate_tool_policy(policy)
    metadata = dict(metadata or {})
    strategy = policy["strategy"]
    models = list(policy.get("models", []))
    fallback_used = evaluator_used = aggregation_used = False
    logger.info("[Tool Policy] strategy=%s models=%s", strategy, models)

    def call(name: str, task: str = prompt, call_image: Any = image) -> str:
        return str(model_call(name, task, call_image, metadata)).strip()

    def accepted(answer: str) -> bool:
        nonlocal evaluator_used
        evaluator = policy.get("evaluator")
        if not evaluator:
            return bool(answer)
        evaluator_used = True
        return call(evaluator, _evaluation_prompt(prompt, answer), None).upper() == "YES"

    if strategy == "single":
        result = call(models[0])
    elif strategy == "cascade":
        result = ""
        attempts: list[str] = []
        for index, name in enumerate(models):
            fallback_used = index > 0
            try:
                candidate = call(name)
            except Exception:
                logger.warning("[Tool Policy] model=%s failed; trying next model", name)
                attempts.append("")
                continue
            if policy.get("stop_condition") == "non_empty" and candidate:
                result = candidate
                break
            if accepted(candidate):
                result = candidate
                break
            attempts.append(candidate)
        if not result and attempts:
            result = attempts[-1]
    elif strategy in {"parallel_first", "parallel_aggregate"}:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(models))
        try:
            futures = {pool.submit(call, name): name for name in models}
            if strategy == "parallel_first":
                result = ""
                for future in concurrent.futures.as_completed(futures):
                    candidate = future.result()
                    if policy.get("stop_condition", "first_complete") == "first_complete" or accepted(candidate):
                        result = candidate
                        break
            else:
                candidates = [future.result() for future in futures]
                aggregation_used = True
                aggregate_prompt = policy.get("aggregation_prompt") or "Combine the candidate answers into one accurate, concise answer."
                result = call(policy["aggregator"], aggregate_prompt + "\n\n" + "\n\n".join(candidates), None)
        finally:
            pool.shutdown(wait=strategy == "parallel_aggregate", cancel_futures=True)
    else:
        condition = policy["condition"]
        if isinstance(condition, str):
            decision = bool(metadata.get(condition))
        elif isinstance(condition, Mapping):
            decision = metadata.get(condition.get("key")) == condition.get("equals")
        else:
            raise ToolPolicyError("conditional condition must be a metadata key or object")
        result = execute_tool_policy(
            policy["if_true"] if decision else policy["if_false"], prompt, image,
            model_call, metadata=metadata,
        )
    logger.info(
        "[Tool Policy] completed strategy=%s models=%s fallback=%s evaluator=%s aggregation=%s",
        strategy, models, fallback_used, evaluator_used, aggregation_used,
    )
    return result
