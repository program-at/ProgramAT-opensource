# ProgramAT Copilot instructions

ProgramAT tools are Python files in `tools/`. The backend executes them with a
camera image and speaks the returned value.

Each take-photo tool must expose `main(image, input_data)` with exactly two parameters.
`image` is an OpenCV BGR array and `input_data` is a dictionary. Return a concise,
audio-friendly string (or the established `audio`/`text` dictionary shape). Do
not print results, connect to the backend, or use WebSockets.

## Take-photo tools

ProgramAT has two visual-context contracts:

- **Static** tools answer from the current frame alone. Use
  `EXECUTION_MODE = "take_photo"`. Static tools may run once in Take Photo or
  repeatedly in Streaming; every streaming invocation receives one selected
  current frame and must be independent. Never add `VIDEO_CONFIG`, a rolling
  window, or cross-frame state to a static tool.
- **Temporal** tools require evidence across multiple moments: recent history,
  sequence, duration, an early/late or before/after comparison, a state change,
  or “what just happened.” Use
  `EXECUTION_MODE = "hosted_video_streaming"`. Temporal tools are streaming-only
  and must declare literal `VIDEO_CONFIG` settings for `window_seconds`,
  `interval_seconds`, `minimum_span_seconds`, and `minimum_unique_frames`.

Do not classify a tool as temporal unless the user's requested answer clearly
depends on multiple moments. Recognition, identification, OCR, description,
classification, and other questions answerable from one frame are static even
when the user may choose to run them continuously.

Define exactly one `TOOL_PROMPT` and use this shape:

```python
from litellm_utils import call_take_photo_vlm

TOOL_NAME = "tool_name"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "One task-specific instruction."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
```

Before writing `TOOL_PROMPT`, analyze whether the requested task is achievable
as one operation or contains genuinely dependent visual or reasoning
subproblems. Default to the simpler prompt. A task that can be completed in one
operation does not need steps, even if the issue description is long or contains
several output-format requirements. Do not create steps merely to restate Task,
Expected output, and Constraints / examples.

Author the shortest high-quality fused prompt that preserves those issue fields.
Include an unavailable-information fallback and request only an accessible,
concise, audio-friendly final answer.

- If the task can be achieved in one operation, write one direct instruction
  with no sequence or numbered steps. This includes simple recognition, OCR,
  classification, and identification.
- For a complex task, when a later conclusion depends on earlier visual
  findings, put one concise ordered sequence of sub-tasks inside the single
  fused prompt. The sequence may use numbered instructions when that improves
  reliability. Ask the VLM to return only the final user-facing answer, not its
  intermediate reasoning.

These are prompt-level instructions for one VLM helper call. Never turn them
into runtime stages, multiple model calls, router stages, specialist calls, or
verification calls. You may consult repository capability descriptions as
reasoning examples, but do not emit capability names, routing metadata,
cascades, or evaluators.

### Prompt examples

Simple task—use one direct instruction with no steps:

```text
Identify the visible hand gesture. Return only the gesture name; if no gesture
is clear, say "No clear gesture."
```

Complex task—use one fused prompt with dependent ordered sub-tasks:

```text
Follow this sequence using the same image:
1. Identify chairs, benches, or other seating.
2. Determine which visible seats are unoccupied.
3. Select the nearest suitable option.
4. Give concise spoken guidance toward it.
If none is visible, say so. Return only the final guidance.
```

Make exactly one VLM helper call. Do not add planner, router, specialist,
fallback model, verification, or provider calls. The shared helper owns the
experiment's model execution.

## Streaming tools

All streaming tools use hosted NVIDIA video through the shared runtime. Keep generated
tools declarative and put event/scene behavior entirely in `TOOL_PROMPT`.

### Hosted-video streaming tools

Use `EXECUTION_MODE = "hosted_video_streaming"` only when the requested result
requires recent history or comparison across time. Generate only `TOOL_NAME`,
that execution mode, required literal `VIDEO_CONFIG`, optional literal
`OUTPUT_CONFIG`, and a task-specific `TOOL_PROMPT`. The prompt must explain the
chronological, early/late, before/after, duration, sequence, or state-change
evidence the VLM should inspect. The shared
runtime supplies the rolling buffer, MP4 encoding, hosted request, event filtering,
deduplication, and output. Never implement RTSP/FFmpeg, frame buffering, async
loops, provider calls, card parsing, before/after state, or take-photo imports.

Supported explicit modes are `take_photo` and `hosted_video_streaming`.
Never infer execution mode from the filename. Do not generate a `main` function for a hosted-video
streaming tool. Never add FFmpeg, frame buffers, HTTP/provider calls, async loops,
or inference logic to generated tools.

## General conventions

- Tools never import other tool modules. Shared utilities such as
  `litellm_utils` are allowed.
- Guard a missing image and catch errors with an audio-friendly error response.
- Avoid GPU-only dependencies unless the issue explicitly requires one.
- Reuse existing non-model utility patterns where appropriate.
- Keep changes scoped to the requested tool and add a focused backend test.
