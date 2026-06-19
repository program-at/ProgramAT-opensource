"""Shared prompt for parser-owned capability and stage decomposition."""

from __future__ import annotations

from typing import Any, Dict

from model_router import load_capability_profiles


def _format_capability_profiles(profiles: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for name in sorted(profiles):
        profile = profiles[name]
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


def build_stage_decomposition_prompt(task_text: str) -> str:
    """Build the standalone routing and stage-decomposition prompt."""
    profiles = load_capability_profiles()
    allowed = ", ".join(sorted(profiles))
    capability_profiles_text = _format_capability_profiles(profiles)
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
    "stages": [
      {{
        "stage_name": "Stage 1",
        "goal": "Complete the user's requested task.",
        "capability": "general_reasoning",
        "input": "Original user request and current input.",
        "expected_output": "A concise result for the user.",
        "preferred_model_type": "general_model"
      }}
    ]
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
- First determine the capability distribution. Then separately decide whether the user request needs multiple parser-owned stages.
- Task decomposition and model selection are separate concerns.
- The LLM parser decides whether multiple stages are required and must explicitly describe them.
- The model router only selects the most appropriate model for a capability. It must not execute stages, manage workflows, pass outputs between stages, or orchestrate pipelines.
- Always return at least one stage. For a single-stage task, return exactly one stage and set should_chain=false.
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
- Stage capabilities must use the same allowed task names. Each stage must include stage_name, goal, capability, input, expected_output, and preferred_model_type. preferred_model_type should describe a model type, not a specific model name.
- Generated tools should execute stages sequentially and pass intermediate outputs between stages.

If should_chain=true, use this stage shape:
{{"stage_name": "Stage 1", "goal": "...", "capability": "object_detection", "input": "", "expected_output": "...", "preferred_model_type": "fast_detector"}}

User request:
{task_text}
"""


__all__ = ["build_stage_decomposition_prompt"]
