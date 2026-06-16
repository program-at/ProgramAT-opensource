"""Tool-facing client for ProgramAT Copilot-routed LLM calls."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import model_router  # noqa: E402


def route_task(task_text: str) -> Dict[str, Any]:
    return model_router.select_model(task_text)


copilot_llm_call = model_router.copilot_llm_call
select_model = model_router.select_model
rank_models = model_router.rank_models
compute_capability_weights = model_router.compute_capability_weights


__all__ = [
    "route_task",
    "copilot_llm_call",
    "select_model",
    "rank_models",
    "compute_capability_weights",
]
