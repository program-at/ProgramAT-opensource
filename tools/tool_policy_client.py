"""Tool-facing client for ProgramAT policy-executed model calls."""

from __future__ import annotations

import sys
from pathlib import Path
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import tool_policy_runtime  # noqa: E402


copilot_llm_call = tool_policy_runtime.copilot_llm_call

__all__ = ["copilot_llm_call"]
