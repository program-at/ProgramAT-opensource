ProgramAT tools are Python files in `tools/`. The backend executes them with a
camera image and speaks the returned value.

Each tool must expose `main(image, input_data)` with exactly two parameters.
`image` is an OpenCV BGR array and `input_data` is a dictionary. Return a concise,
audio-friendly string (or the established `audio`/`text` dictionary shape). Do
not print results, connect to the backend, or use WebSockets.

## Take-photo tools

Use the issue's `Task Stages` as the implementation plan. Import the existing
tool-facing call:

```python
from model_router_client import copilot_llm_call
```

Generate exactly one ordered call per declared stage. Use the stage capability
and preserve its goal as the step-specific `goal` or user message. Pass
`images=[image]` whenever the stage needs the scene. Later calls must explicitly
receive useful earlier `artifact` values in `metadata` and/or messages. Return
the last call's `response`. A one-stage issue produces one call. Never fuse a
multi-stage issue into one `TOOL_PROMPT`, choose an implementation, add retries,
or add evaluation/escalation logic; the backend supplies one fixed model per call.

## Streaming tools

Preserve the repository's existing streaming patterns and cadence. Keep output
to about 15 spoken words and return an empty string when nothing useful changed.
Do not alter NVIDIA hosted streaming or RTVI code while implementing a tool.

## General conventions

- Tools never import other tool modules. Shared utilities such as
  `litellm_utils` are allowed.
- Guard a missing image and catch errors with an audio-friendly error response.
- Avoid GPU-only dependencies unless the issue explicitly requires one.
- Reuse existing non-model utility patterns where appropriate.
- Keep changes scoped to the requested tool and add a focused backend test.
