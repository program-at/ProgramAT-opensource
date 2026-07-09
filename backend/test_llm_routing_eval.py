#!/usr/bin/env python3
"""Sample Gemini semantic-decomposition eval on selected task-corpus entries.

This script:
1. Picks tasks from route_task_batch.TASKS by index.
2. Calls the existing system LLM once per task to get an ordered semantic stage list.
3. Records the ordered execution-policy candidates for each planned capability.
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
from model_router import system_llm_call
from route_task_batch import TASKS
from stage_decomposition import build_stage_decomposition_prompt, normalize_stage_plan


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BACKEND_DIR / "gemini_routing_eval_results.sample.json"

def build_prompt(task_text: str) -> str:
    """Compatibility entrypoint for the shared production decomposition prompt."""
    return build_stage_decomposition_prompt(task_text)




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
        messages=[{"role": "user", "content": prompt}],
    )
    raw = extract_text(response)
    parsed = parse_json_text(raw)
    planner_output = normalize_stage_plan(parsed, source_task=task)
    stages = planner_output["stages"]
    capability_sequence = [
        stage["capability"] for stage in stages
    ]
    policies = model_router.load_execution_policies()

    row = {
        "task": task,
        "raw_response": raw,
        "planner_output": planner_output,
        "execution_policies": {
            capability: policies.get(capability, [])
            for capability in capability_sequence
        },
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
            capabilities = [stage["capability"] for stage in result["planner_output"]["stages"]]
            print(f"  -> capabilities: {', '.join(capabilities)}")
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
