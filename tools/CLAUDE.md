# tools/ — executable assistive tools

Follow `.github/copilot-instructions.md` as the generated-tool contract. New and updated tools normally define `TOOL_NAME`, one shared `TOOL_PROMPT`, and both Take Photo and Streaming lifecycle hooks:

```python
async def on_take_photo(runtime, image, input_data): ...
async def on_stream_start(runtime, input_data): ...
async def on_frame(runtime, frame): ...
async def on_stream_stop(runtime): ...
```

The tool owns model choices, frame selection, temporal logic, orchestration, state, and output timing. Use approved model helpers and runtime frames, state, cancellation, and emission. Keep output concise and accessible.

Do not access credentials, environment variables, provider SDKs, arbitrary networks, WebSockets, subprocesses, unrestricted filesystem writes, other sessions, or backend-private globals.

Old `main(image, input_data)` and declarative `EXECUTION_MODE`/`TOOL_POLICY` tools remain supported for compatibility, but do not use those contracts for new or updated tools.
