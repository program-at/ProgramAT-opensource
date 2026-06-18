#!/usr/bin/env python3
"""Sample Gemini routing-analysis eval on selected route_task_batch tasks.

This script:
1. Picks tasks from route_task_batch.TASKS by index.
2. Calls the existing system LLM once per task to get routing_analysis and pipeline_analysis.
3. Runs centralized model_router.select_model(task, routing_analysis=..., pipeline_analysis=...).
4. Sleeps between requests to reduce rate-limit risk.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import model_router
from litellm_utils import extract_text
from model_router import load_capability_profiles, system_llm_call
from route_task_batch import TASKS


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BACKEND_DIR / "gemini_routing_eval_results.sample.json"

CAPABILITY_PROFILES = load_capability_profiles()
ALLOWED_TASK_NAMES = sorted(CAPABILITY_PROFILES.keys())


def format_capability_profiles() -> str:
    lines = []
    for name in ALLOWED_TASK_NAMES:
        profile = CAPABILITY_PROFILES[name]
        lines.append(f"- {name}")
        if profile.get("description"):
            lines.append(f"  description: {profile['description']}")
        if profile.get("include_examples"):
            lines.append("  includes:")
            lines.extend(f"    - {example}" for example in profile["include_examples"])
        if profile.get("exclude_examples"):
            lines.append("  excludes:")
            lines.extend(f"    - {example}" for example in profile["exclude_examples"])
        if profile.get("notes"):
            lines.append(f"  notes: {profile['notes']}")
    return "\n".join(lines)


def build_prompt(task_text: str) -> str:
    allowed = ", ".join(ALLOWED_TASK_NAMES)
    capability_profiles_text = format_capability_profiles()
    return f"""You are a routing-analysis assistant for a visual assistive technology backend.

Analyze this user request and return ONLY a valid JSON object with this exact schema:
{{
  "routing_analysis": {{
    "tasks": [
            {{"name": "ocr", "weight": 0.8, "reason": "..."}},
            {{"name": "general_reasoning", "weight": 0.2, "reason": "..."}}
    ],
        "latency_sensitivity": {{"level": "high|medium|low", "weight": 0.8, "reason": "..."}}
  }},
  "pipeline_analysis": {{
    "should_chain": false,
    "reason": "...",
    "artifact_value_score": 0.0,
    "information_reduction": "none|minor|moderate|significant",
    "artifact_gain": 0.0,
    "complexity_penalty": 0.0,
    "pipeline_score": 0.0,
    "stages": []
  }}
}}

Rules:
- Use 2-4 task entries when possible.
- Use only these task names: {allowed}
- Select only from the provided capability names.
- Do not create new capability names.
- Map the user request onto the closest existing capabilities.
- Prefer the minimum necessary capability set.
- Prefer 1 capability when one capability is sufficient.
- Prefer 2 capabilities over 3 unless the task truly requires more.
- Do not include a capability unless removing it would significantly reduce task success.
- Use capability exclusions as hard negative evidence.
- You are given the full capability definitions below. Use them to decide what is required.

Capability definitions:
{capability_profiles_text}

- Task weights should sum to approximately 1.0.
- high latency_sensitivity for live camera, navigation, object finding, real-time assistive feedback.
- medium latency_sensitivity for normal interactive tasks.
- low latency_sensitivity for non-urgent/offline/codegen tasks.
- First determine the capability distribution. Then separately decide whether those capabilities should run in one model or as a pipeline.
- Do not chain simply because multiple capabilities exist.
- Do NOT decide should_chain=false because a single modern VLM could theoretically perform all capabilities. That is not the criterion.
- Decide based on information reduction: can an earlier stage produce a structured intermediate artifact that reduces the search space, visual scope, or reasoning burden for a later stage?
- Useful intermediate artifacts include bounding boxes, object coordinates, segmentation masks, cropped regions, OCR text, object lists, target locations, clock-face directions, and navigation waypoints.
- Increase chaining confidence for useful transitions: object_detection -> spatial_relationship, object_detection -> navigation, object_detection -> general_reasoning, ocr -> general_reasoning, ocr -> map_web, spatial_relationship -> navigation, map_web -> general_reasoning.
- Prefer should_chain=true for "find X then guide me", "find X then localize Y", "detect X then reason about X", and "extract text then analyze text".
- Prefer should_chain=false for holistic tasks like room description, scene description, chart explanation, outfit evaluation, and image summarization when no meaningful artifact reduces later work.
- Default to should_chain=false only when no useful intermediate artifact or information reduction exists.
- Include artifact_value_score from 0.0 to 1.0 and information_reduction as none/minor/moderate/significant.
- Do not hardcode a maximum stage count. Infer the stage count dynamically from useful artifact flow.
- Optimize for the minimum useful stage count, not the minimum stage count alone.
- Use a complexity penalty to discourage unnecessary stages:
  pipeline_score = artifact_gain - complexity_penalty
  complexity_penalty = (stage_count - 1) * penalty_per_stage
- Add another stage when it preserves an important intermediate artifact or materially improves information reduction/task success probability.
- Do not add another stage when it only repeats work, changes wording, or does not improve downstream inputs.
- Stage capabilities must use the same allowed task names. preferred_model_type should describe a model type, not a specific model name.

If should_chain=true, use this stage shape:
{{"capability": "object_detection", "purpose": "...", "input": "...", "output": "...", "preferred_model_type": "fast_detector"}}

User request:
{task_text}
"""


def parse_json_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def pick_tasks(indices: List[int]) -> List[Dict[str, Any]]:
    picked = []
    for idx in indices:
        if idx < 1 or idx > len(TASKS):
            raise ValueError(f"Index out of range: {idx}. Valid range: 1-{len(TASKS)}")
        picked.append({"index": idx, "task": TASKS[idx - 1]})
    return picked


def eval_task(task: str, sleep_seconds: float) -> Dict[str, Any]:
    prompt = build_prompt(task)
    response = system_llm_call(
        capability="text_parse",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = extract_text(response)
    parsed = parse_json_text(raw)
    routing_analysis = parsed.get("routing_analysis") if isinstance(parsed, dict) else None
    pipeline_analysis = parsed.get("pipeline_analysis") if isinstance(parsed, dict) else None

    route_result = model_router.select_model(
        task,
        routing_analysis=routing_analysis,
        pipeline_analysis=pipeline_analysis,
    )

    row = {
        "task": task,
        "raw_response": raw,
        "routing_analysis": route_result.get("routing_analysis"),
        "pipeline_analysis": route_result.get("pipeline_analysis"),
        "selected_model": route_result.get("selected_model"),
        "final_execution_plan": route_result.get("final_execution_plan"),
        "stage_model_selection": [
            {
                "index": x.get("index"),
                "capability": x.get("capability"),
                "selected_model": x.get("selected_model"),
            }
            for x in route_result.get("stage_model_selection", [])
        ],
        "top3": [
            {
                "model": x.get("model"),
                "final_score": round(float(x.get("final_score", 0.0)), 4),
                "task_match_score": round(float(x.get("task_match_score", x.get("capability_score", 0.0))), 4),
                "latency_match_score": round(float(x.get("latency_match_score", 0.0)), 4),
            }
            for x in route_result.get("ranking", [])[:3]
        ],
        "capability_count": len(route_result.get("routing_analysis", {}).get("tasks", [])) if route_result.get("routing_analysis") else 0,
    }

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[1, 27, 43],
        help="1-based task indices from route_task_batch.TASKS",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep duration between Gemini calls",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path",
    )
    args = parser.parse_args()

    selected = pick_tasks(args.indices)
    rows = []
    errors = []

    for item in selected:
        idx = item["index"]
        task = item["task"]
        print(f"[{idx}] Evaluating: {task}")
        try:
            result = eval_task(task, args.sleep_seconds)
            result["index"] = idx
            rows.append(result)
            print(f"  -> selected_model: {result['selected_model']}")
        except Exception as exc:
            errors.append({"index": idx, "task": task, "error": str(exc)})
            print(f"  -> ERROR: {exc}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "indices": args.indices,
        "sleep_seconds": args.sleep_seconds,
        "result_count": len(rows),
        "error_count": len(errors),
        "results": rows,
        "errors": errors,
    }

    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote results to: {args.out}")


if __name__ == "__main__":
    main()
