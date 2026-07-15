ProgramAT tools are Python files in `tools/`. The backend executes them with a
camera image and speaks the returned value.

Each tool must expose `main(image, input_data)` with exactly two parameters.
`image` is an OpenCV BGR array and `input_data` is a dictionary. Return a concise,
audio-friendly string (or the established `audio`/`text` dictionary shape). Do
not print results, connect to the backend, or use WebSockets.

## Take-photo tools

Define exactly one `TOOL_PROMPT` and use this shape:

```python
from litellm_utils import call_take_photo_baseline_vlm

TOOL_NAME = "tool_name"
TOOL_PROMPT = "One task-specific instruction."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_baseline_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
```

Read `Prompt strategy` in the issue. For `no_planner`, copy `P1 exact prompt`
verbatim. For `copilot_fused_prompt`, author the shortest high-quality fused
prompt that preserves Task, Expected output, and Constraints / examples. Include
an unavailable-information fallback and request only an accessible, concise,
audio-friendly final answer. Simple recognition, OCR, classification, and
identification tasks should be direct, without steps. Use an ordered logical
sequence only when later reasoning depends on earlier visual findings. You may
consult repository capability descriptions as reasoning examples, but do not
emit capability names, runtime stages, routing metadata, cascades, or evaluators.

Simple example: `Identify the visible hand gesture. Return only the gesture name; if no gesture is clear, say "No clear gesture."`

Complex example: `Inspect the image for chairs, benches, or other seating. Determine which seats are visibly unoccupied, select the nearest suitable option, and give concise spoken guidance toward it. If none is visible, say so. Return only the final guidance.`

Make exactly one helper call. Do not add planner, router, specialist, fallback
model, verification, or provider calls. The shared helper owns Gemini Flash Lite.

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
