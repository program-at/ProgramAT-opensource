"""Tool-facing client for ProgramAT Copilot-routed LLM calls."""

from __future__ import annotations

import sys
from pathlib import Path
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import model_router  # noqa: E402


copilot_llm_call = model_router.copilot_llm_call

__all__ = ["copilot_llm_call"]
