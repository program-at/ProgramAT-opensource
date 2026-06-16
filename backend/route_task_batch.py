"""Run a fixed batch of user tasks through the semantic model router."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import model_router


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_OUT = BACKEND_DIR / "model_router_task_results.json"
DEFAULT_MD_OUT = BACKEND_DIR / "model_router_task_results.md"

TASKS = [
    "Read the medication bottle label.",
    "Read the expiration date.",
    "Read a restaurant menu.",
    "Read a street sign.",
    "Read the ingredients on a food package.",
    "Read a handwritten note.",
    "Read the text on a computer screen.",
    "Read the instructions on an appliance.",
    "Find my cup.",
    "Find my phone.",
    "Locate my keys.",
    "Find the remote control.",
    "Find the door handle.",
    "Find an empty chair.",
    "Find the elevator button.",
    "Find the traffic light.",
    "Locate the correct medication bottle.",
    "Find the nearest trash can.",
    "Describe the position of the chair relative to the table.",
    "Tell me whether the cup is to the left or right of the bottle.",
    "Describe where the door is.",
    "Tell me how far away the stairs are.",
    "Describe obstacles in front of me.",
    "Explain the layout of the room.",
    "Tell me whether the chair is behind the desk.",
    "Describe the distance between me and the crosswalk.",
    "Guide me to the building entrance.",
    "Help me find the nearest exit.",
    "Guide me to an available seat.",
    "Navigate me to the checkout counter.",
    "Help me reach the elevator.",
    "Guide me to the correct bus stop.",
    "Help me cross the street safely.",
    "Navigate me through a crowded hallway.",
    "Describe what is happening in front of me.",
    "Explain the scene around me.",
    "Tell me whether it is safe to cross the street.",
    "Describe the activity in the room.",
    "Explain what the people are doing.",
    "Tell me whether this outfit matches.",
    "Describe the overall environment.",
    "Explain whether the stove appears to be on.",
    "Interpret this map.",
    "Explain this chart.",
    "Read the schedule on this screen.",
    "Interpret a bus timetable.",
    "Explain a webpage layout.",
    "Read a departure board.",
    "Interpret a weather chart.",
    "Read a calendar.",
    "Tell me whether I am moving toward the object.",
    "Guide my camera toward the target.",
    "Help me center the object in view.",
    "Tell me how to adjust the camera.",
    "Explain whether I am getting closer to the door.",
    "Guide me to align with the crosswalk.",
    "Summarize what happened in this video.",
    "Describe the actions occurring in the video.",
    "Explain the sequence of events.",
    "Track a person's activities.",
    "Describe changes in the environment.",
    "Summarize the interaction between people.",
    "Find the passenger door handle.",
    "Find my Uber.",
    "Locate the correct medication bottle.",
    "Guide me to an empty seat.",
    "Tell me whether it is safe to cross the street.",
    "Help me find the checkout counter.",
    "Find the elevator and guide me to it.",
    "Locate the restroom entrance.",
    "Find the nearest available parking space.",
    "Identify the correct bus and tell me where the entrance is.",
]


def _rounded_weights(weights: Dict[str, float]) -> Dict[str, float]:
    return {
        capability: round(value, 4)
        for capability, value in sorted(weights.items(), key=lambda item: item[1], reverse=True)
        if value > 0.0001
    }


def _top_models(ranking: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "final_score": round(float(row["final_score"]), 4),
            "capability_score": round(float(row["capability_score"]), 4),
            "latency_ms": row["latency_ms"],
        }
        for row in ranking[:limit]
    ]


def run_batch() -> Dict[str, Any]:
    rows = []
    for index, task in enumerate(TASKS, start=1):
        result = model_router.select_model(task)
        rows.append({
            "index": index,
            "task": task,
            "selected_model": result["selected_model"],
            "capability_weights": _rounded_weights(result["capability_weights"]),
            "top_models": _top_models(result["ranking"]),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(rows),
        "rows": rows,
    }


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Model Router Task Results",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "| # | Task | Selected model | Top capability weights | Top 3 models |",
        "|---:|---|---|---|---|",
    ]
    for row in payload["rows"]:
        weights = ", ".join(
            f"{capability}={value:.4f}"
            for capability, value in row["capability_weights"].items()
        )
        top_models = ", ".join(
            f"{model['model']} ({model['final_score']:.4f})"
            for model in row["top_models"]
        )
        task = row["task"].replace("|", "\\|")
        lines.append(
            f"| {row['index']} | {task} | {row['selected_model']} | {weights} | {top_models} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    payload = run_batch()
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.md_out)

    print(f"Wrote {payload['task_count']} rows to {args.json_out}")
    print(f"Wrote Markdown summary to {args.md_out}")


if __name__ == "__main__":
    main()
