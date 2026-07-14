ProgramAT tools are Python files in `tools/`. The backend executes them with a
camera image and speaks the returned value.

Each tool must expose `main(image, input_data)` with exactly two parameters.
`image` is an OpenCV BGR array and `input_data` is a dictionary. Return a concise,
audio-friendly string (or the established `audio`/`text` dictionary shape). Do
not print results, connect to the backend, or use WebSockets.

## Take-photo tools

Implement one user-facing task with one concise task-specific prompt:

```python
from litellm_utils import call_take_photo_baseline_vlm

TOOL_PROMPT = "Describe exactly what this tool should determine from the image."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    answer = call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
    return answer
```

Make exactly one `call_take_photo_baseline_vlm` call and return its answer
directly. Do not add any other model calls, specialist calls, fallback logic,
verification passes, or provider SDKs. Do not hardcode a model name in the tool;
the shared helper owns the fixed Gemini Flash Lite model.

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
