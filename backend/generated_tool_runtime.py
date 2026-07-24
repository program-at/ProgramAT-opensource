"""Minimal execution resources for Copilot-generated tool modules."""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolFrame:
    """One decoded camera frame and its transport metadata."""

    frame_id: int
    timestamp: float
    image: Any
    width: int
    height: int
    image_base64: str = ""


class FrameStore:
    """Bounded frame storage; selection decisions belong to generated tools."""

    def __init__(self, max_frames: int = 90):
        self._frames: Deque[ToolFrame] = deque(maxlen=max(1, int(max_frames)))

    def add(self, frame: ToolFrame) -> None:
        self._frames.append(frame)

    def latest(self) -> Optional[ToolFrame]:
        return self._frames[-1] if self._frames else None

    def recent(
        self, count: Optional[int] = None, seconds: Optional[float] = None
    ) -> list[ToolFrame]:
        frames = list(self._frames)
        if seconds is not None and frames:
            cutoff = frames[-1].timestamp - max(0.0, float(seconds))
            frames = [frame for frame in frames if frame.timestamp >= cutoff]
        if count is not None:
            frames = frames[-max(0, int(count)):]
        return frames

    def clear(self) -> None:
        self._frames.clear()


EmitCallback = Callable[[Dict[str, Any]], Awaitable[bool]]


class ToolRuntime:
    """Infrastructure-only API exposed to one tool session."""

    def __init__(
        self,
        *,
        client_id: str,
        tool_name: str,
        tool_version: str,
        session_id: str,
        request_id: Optional[str] = None,
        frame_store: Optional[FrameStore] = None,
        emit_callback: Optional[EmitCallback] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.client_id = client_id
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.session_id = session_id
        self.current_request_id = request_id or secrets.token_hex(16)
        self._frames = frame_store or FrameStore()
        self._emit_callback = emit_callback
        self._cancelled = cancelled or (lambda: False)
        self._state: Dict[str, Any] = {}
        self._emit_index = 0

    @property
    def current_frame(self) -> Optional[ToolFrame]:
        return self._frames.latest()

    @property
    def recent_frames(self) -> list[ToolFrame]:
        return self._frames.recent()

    def get_latest_frame(self) -> Optional[ToolFrame]:
        return self._frames.latest()

    def get_recent_frames(
        self, count: Optional[int] = None, seconds: Optional[float] = None
    ) -> list[ToolFrame]:
        return self._frames.recent(count=count, seconds=seconds)

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def clear_state(self) -> None:
        self._state.clear()

    def is_cancelled(self) -> bool:
        return bool(self._cancelled())

    async def emit(
        self,
        text: Any,
        *,
        partial: bool = False,
        final: bool = False,
        replace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.is_cancelled() or self._emit_callback is None:
            return False
        self._emit_index += 1
        event = {
            "request_id": self.current_request_id,
            "invocation_id": self.current_request_id,
            "tool_name": self.tool_name,
            "session_id": self.session_id,
            "result_index": self._emit_index,
            "text": "" if text is None else str(text).strip(),
            "partial": bool(partial),
            "final": bool(final),
            "replace": bool(replace),
            "metadata": dict(metadata or {}),
        }
        accepted = await self._emit_callback(event)
        logger.info(
            "[GeneratedTool] emit tool=%s session=%s request=%s index=%d "
            "partial=%s final=%s accepted=%s",
            self.tool_name,
            self.session_id,
            self.current_request_id,
            self._emit_index,
            bool(partial),
            bool(final),
            bool(accepted),
        )
        return bool(accepted)


def has_executable_lifecycle(tool_code: str) -> bool:
    """Detect lifecycle functions without executing the module."""
    import ast

    try:
        tree = ast.parse(tool_code or "")
    except SyntaxError:
        return False
    lifecycle_names = {
        "on_take_photo", "on_stream_start", "on_frame", "on_stream_stop"
    }
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in lifecycle_names
        for node in tree.body
    )


def load_generated_tool(
    tool_code: str, *, filename: str, injected_globals: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute validated generated source once for one isolated tool session."""
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": filename,
        "__name__": f"generated_tool_{secrets.token_hex(8)}",
    }
    namespace.update(injected_globals or {})
    compiled = compile(tool_code, filename, "exec")
    exec(compiled, namespace, namespace)
    return namespace


async def invoke_tool_hook(namespace: Dict[str, Any], name: str, *args) -> Any:
    """Invoke one generated lifecycle hook, accepting sync or async functions."""
    hook = namespace.get(name)
    if not callable(hook):
        return None
    result = hook(*args)
    if inspect.isawaitable(result):
        return await result
    return result


async def cancel_tasks(tasks: Iterable[asyncio.Task]) -> None:
    pending = [task for task in tasks if task is not None and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


__all__ = [
    "FrameStore",
    "ToolFrame",
    "ToolRuntime",
    "cancel_tasks",
    "has_executable_lifecycle",
    "invoke_tool_hook",
    "load_generated_tool",
]
