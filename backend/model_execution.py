"""Public generated-tool boundary for explicit model execution policies.

Generated tools declare policy data; this module validates and executes it through
the centralized runtime. Provider adapters and orchestration stay internal.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from tool_policy_runtime import execute_resolved_tool_policy


def execute_tool_policy(
    *, image: Any, prompt: str, policy: Any = None,
    tool_name: Optional[str] = None, mode: str = "take_photo",
    request_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Validate and execute exactly the declared policy, or the compatibility default."""
    execution_metadata = dict(metadata or {})
    execution_metadata.update({"mode": mode, "tool_name": tool_name or ""})
    return execute_resolved_tool_policy(
        prompt=prompt,
        image=image,
        policy=policy,
        request_id=request_id,
        metadata=execution_metadata,
    )


__all__ = ["execute_tool_policy"]
