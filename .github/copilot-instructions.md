ProgramAT tools are Python files in `tools/`. The backend executes them with a
camera image and speaks the returned value.

Each tool must expose `main(image, input_data)` with exactly two parameters.
`image` is an OpenCV BGR array and `input_data` is a dictionary. Return a concise,
audio-friendly string (or the established `audio`/`text` dictionary shape). Do
not print results, connect to the backend, or use WebSockets.

## Take-photo tools

When the issue's `Fused VLM Prompt` section is populated, use it verbatim:

```python
from litellm_utils import call_take_photo_baseline_vlm

FUSED_VLM_PROMPT = "Copy the exact text from the issue's Fused VLM Prompt section."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    answer = call_take_photo_baseline_vlm(image=image, prompt=FUSED_VLM_PROMPT)
    return answer
```

Make exactly one `call_take_photo_baseline_vlm` call and return its answer
directly. Do not author, rewrite, or decompose the prompt. Do not add any other
model calls, planner or router calls, stages, specialist calls, fallback logic,
verification passes, or provider SDKs. Do not hardcode a model name in the tool;
the shared helper owns the fixed Gemini Flash Lite model.

When that issue section is empty, preserve the existing P1 behavior: define one
concise task-specific `TOOL_PROMPT` and pass it to the same helper exactly once.

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
