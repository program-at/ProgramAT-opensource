# ProgramAT Copilot instructions

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
TOOL_DIFFICULTY_START = "gemini_flash_lite"


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_baseline_vlm(
        image=image,
        prompt=TOOL_PROMPT,
        tool_name=TOOL_NAME,
        difficulty_start=TOOL_DIFFICULTY_START,
    )
```

Copy `TOOL_DIFFICULTY_START` exactly from the issue's Difficulty start field.
It is creation-time task metadata; do not classify difficulty from runtime
images and do not make a model call to recompute it.

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

Preserve the repository's existing streaming patterns and cadence. Keep output
to about 15 spoken words and return an empty string when nothing useful changed.
Copy the issue's Difficulty start value into a `TOOL_DIFFICULTY_START` string
constant; the backend reads this creation-time metadata before running the
policy cascade. Never recompute difficulty from a streaming frame.
Do not alter NVIDIA hosted streaming or RTVI code while implementing a tool.

## General conventions

- Tools never import other tool modules. Shared utilities such as
  `litellm_utils` are allowed.
- Guard a missing image and catch errors with an audio-friendly error response.
- Avoid GPU-only dependencies unless the issue explicitly requires one.
- Reuse existing non-model utility patterns where appropriate.
- Keep changes scoped to the requested tool and add a focused backend test.
