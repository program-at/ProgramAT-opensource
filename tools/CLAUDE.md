# tools/ — executable assistive tools

Generated tools own their model choices, prompts, frame selection, temporal
logic, orchestration, and output timing. The backend supplies decoded frames,
isolated state, cancellation, and result transport.

New and updated tools should declare `TOOL_NAME` and
`EXECUTION_MODE = "take_photo"` and may implement:

```python
async def on_take_photo(runtime, image, input_data): ...
async def on_stream_start(runtime, input_data): ...
async def on_frame(runtime, frame): ...
async def on_stream_stop(runtime): ...
```

`frame` has `frame_id`, `timestamp`, `image`, `width`, `height`, and
`image_base64`. The minimal runtime exposes:

```python
runtime.current_request_id
runtime.current_frame
runtime.get_recent_frames(count=None, seconds=None)
runtime.get_state(key, default=None)
runtime.set_state(key, value)
runtime.clear_state()
runtime.is_cancelled()
await runtime.emit(text, partial=False, final=False, replace=False, metadata=None)
```

Use `call_model()` from `litellm_utils` directly for configured LiteLLM models.
Use `call_openai_responses_model()` only for models requiring the Responses API,
and use `extract_text()` to normalize LiteLLM output. Model names must be
explicit in tool code. Inspect `backend/model_registry.py` while authoring;
default to `gemini/gemini-3.1-flash-lite-preview` for ordinary visual tasks.

Tool helpers may use approved libraries such as OpenCV, NumPy, PIL, CLIP, and
ordinary asyncio control flow. Tools decide how to compare frames, buffer
history, suppress unchanged scenes, run models sequentially or concurrently,
and emit one or many results. Check `runtime.is_cancelled()` and tool-owned
scene/version state before emitting work that may have become stale.

Do not access environment variables or API keys, provider SDKs, WebSockets,
arbitrary networks, subprocesses, other sessions, or the filesystem. Do not
modify backend globals. Never provide visual-only navigation landmarks; use
body-relative, tactile, auditory, or stable structural directions.

Old `main(image, input_data)` plus literal `TOOL_POLICY` tools remain supported
temporarily, but do not use that contract for new or updated tools.
