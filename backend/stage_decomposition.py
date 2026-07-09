"""Shared prompt for parser-owned capability and stage decomposition."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from model_router import load_capability_profiles, normalize_capability_name
from moondream_provider import is_card_identification_task


PLAYING_CARD_STAGE_GOAL = "Identify only the playing cards by rank and suit."


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
    """Build the semantic stage-decomposition prompt."""
    profiles = load_capability_profiles()
    allowed = ", ".join(sorted(profiles))
    capability_profiles_text = _format_capability_profiles(profiles)
    return f"""You are a task-decomposition assistant for a visual assistive technology backend.

Analyze this user request and return ONLY a valid JSON object with this exact schema:
{{
  "stages": [
    {{
      "goal": "Complete the user's requested task.",
      "capability": "general_reasoning"
    }}
  ]
}}

Rules:
- Always return at least one stage.
- Return exactly one stage when one semantic operation is sufficient.
- Return multiple stages only when the request contains distinct ordered operations.
- Keep stages in the order they must be completed.
- Use only these capability names: {allowed}
- Do not create new capability names.
- Map each stage goal to the closest capability.
- Use the minimum number of stages that fully represents the request.
- Do not add a stage that merely restates or summarizes another stage.
- Do not add general_reasoning unless the user explicitly requests a semantic operation not covered by another stage.
- Use capability exclusions as hard negative evidence.
- Each stage must contain exactly two fields: goal and capability.
- Do not return any fields other than stages, goal, and capability.
- The execution policy—not this planner—owns implementation details.
- Apply the playing-card rule only when the user request explicitly asks to
  identify playing cards, poker cards, card rank, or card suit.
- For playing-card identification tasks, use this exact stage goal:
  "{PLAYING_CARD_STAGE_GOAL}"

Capability definitions:
{capability_profiles_text}

User request:
{task_text}
"""


def normalize_stage_plan(
    raw_plan: Any,
    supported_capabilities: Optional[Iterable[str]] = None,
    source_task: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the strict semantic planner schema."""
    if not isinstance(raw_plan, dict) or set(raw_plan) != {"stages"}:
        raise ValueError("Planner output must contain only the stages field")
    if not isinstance(raw_plan["stages"], list) or not raw_plan["stages"]:
        raise ValueError("Planner must return a non-empty stages list")

    supported = {
        str(name).strip().lower()
        for name in (supported_capabilities or load_capability_profiles().keys())
        if str(name).strip()
    }
    source_is_card_task = is_card_identification_task(source_task) if source_task is not None else None
    stages = []
    for raw_stage in raw_plan["stages"]:
        if not isinstance(raw_stage, dict) or set(raw_stage) != {"goal", "capability"}:
            raise ValueError("Each planner stage must contain only goal and capability")
        goal = str(raw_stage["goal"]).strip()
        capability = normalize_capability_name(raw_stage["capability"])
        if not goal:
            raise ValueError("Each planner stage must provide a non-empty goal")
        if capability not in supported:
            raise ValueError(f"Planner returned unknown capability: {capability}")
        if is_card_identification_task(goal):
            if source_is_card_task is False:
                continue
            goal = PLAYING_CARD_STAGE_GOAL
        stages.append({"goal": goal, "capability": capability})
    if not stages:
        raise ValueError("Planner returned no stages relevant to the source task")
    return {"stages": stages}


__all__ = ["build_stage_decomposition_prompt", "normalize_stage_plan"]
