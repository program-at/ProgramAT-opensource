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
  `EXECUTION_MODE = "hosted_video_streaming"` to select their Streaming behavior.
  They must declare literal `VIDEO_CONFIG` settings for `window_seconds`,
  `interval_seconds`, `minimum_span_seconds`, and `minimum_unique_frames`.

Take Photo is available for every tool and always supplies exactly one current
image with the tool's normal `TOOL_PROMPT`. For a temporal tool, author that
prompt to work safely with either input shape:

- With multiple chronological frames, use motion, sequence, duration, and state
  changes as relevant to the task.
- With one image, use only recognizable static evidence. Never invent unseen
  movement, earlier state, or later state.
- When one frame is insufficient, return the task-specific concise uncertainty
  response requested by the prompt.

Do not classify a tool as temporal unless the user's requested answer clearly
depends on multiple moments. Recognition, identification, OCR, description,
classification, and other questions answerable from one frame are static even
when the user may choose to run them continuously.

The issue `Mode` value is a parser suggestion, not an instruction to ignore the
request semantics. Read the preserved `ORIGINAL_PROMPTS` and Task/Expected output
before deciding the contract. Correct the suggested mode when they conflict.
If a request permits both static and dynamic examples, choose temporal whenever
correct interpretation may require recent movement or several ordered frames.
For sign language specifically, a current held posture is static; recent hand
movements, a sign lasting seconds, or the sign/phrase just made are temporal.

Define exactly one `TOOL_PROMPT` and use this shape:

```python
from model_execution import execute_tool_policy

TOOL_NAME = "tool_name"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "One task-specific instruction."
TOOL_POLICY = {
    "strategy": "single",
    "models": ["gemini-3.1-flash-lite"],
}


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return execute_tool_policy(
        image=image, prompt=TOOL_PROMPT, policy=TOOL_POLICY, tool_name=TOOL_NAME
    )
```

Copilot is responsible for choosing and emitting a literal `TOOL_POLICY`.
Inspect `backend/model_registry.py` and `backend/strategy_registry.py`, then
choose the simplest policy that satisfies the request. For ordinary requests
without a meaningful execution preference or specialized-model need, emit the
single Gemini 3.1 Flash Lite policy shown above. The backend only validates and
executes this data; it never infers difficulty, chooses models, or expands it.

- For speed, prefer one fast suitable model: YOLO only for supported object
  detection, Moondream only for very simple low-risk tasks where reduced
  accuracy is acceptable, and otherwise Gemini 3.1 Flash Lite.
- For accuracy when substantially higher latency is acceptable, prefer GPT-5.
- For "usually fast, but more accurate when needed," use `conditional` or
  `cascade`, starting with Gemini 3.1 Flash Lite and escalating to GPT-5 only
  under an explicit sufficiency condition.
- Use `cascade` only for an explicit fast-first, stronger-fallback request.
- Use `parallel_first` only when multiple models should start together and the
  first acceptable result should win.
- Use `parallel_aggregate` only to compare or combine several outputs.
- Use `conditional` only for a clearly defined fast-default/stronger condition.
- Gemini 3.1 Flash Lite is the default general VLM. YOLO is object detection
  only. Moondream is for simple, low-risk tasks where lower accuracy is
  acceptable. GPT-5 is for accuracy/strong reasoning despite latency and cost.
  GPT-4o-mini is mainly an evaluator or aggregator.

Policies are static data, never custom routing code. Pass the declaration to
`execute_tool_policy` as `policy=TOOL_POLICY`. Never implement model orchestration outside
`TOOL_POLICY`, and never add cascade, evaluation, parallelism, or aggregation
without a user-driven reason.

```python
TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}
```

```python
TOOL_POLICY = {
    "strategy": "cascade",
    "models": ["moondream", "gemini-3.1-flash-lite", "gpt-5"],
    "evaluator": "gpt-4o-mini",
}
```

```python
TOOL_POLICY = {
    "strategy": "parallel_aggregate",
    "models": ["gemini-3.1-flash-lite", "gpt-5"],
    "aggregator": "gpt-4o-mini",
}
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

These are prompt-level instructions for one shared helper call. Never turn them
into runtime stages, multiple model calls, or custom routing. Any requested
multi-model behavior belongs only in the literal `TOOL_POLICY`.

### Prompt examples

Simple task—use one direct instruction with no steps:

```text
Identify the visible hand gesture. Return only the gesture name; if no gesture
is clear, say "No clear gesture."
```

Complex task—use one fused prompt with dependent ordered sub-tasks:

```python
from model_execution import execute_tool_policy

TOOL_NAME = "empty_chair"
EXECUTION_MODE = "take_photo"
TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}
TOOL_PROMPT = """Follow this sequence using the same image:
1. Identify chairs, benches, or other seating.
2. Determine which visible seats are unoccupied.
3. Select the nearest suitable option.
4. Give concise spoken guidance toward it.
If none is visible, say so. Return only the final guidance."""

def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return execute_tool_policy(
        image=image, prompt=TOOL_PROMPT, policy=TOOL_POLICY, tool_name=TOOL_NAME
    )
```

The empty-chair example uses the default visible policy above: strategy
`single`, model `gemini-3.1-flash-lite`. Make exactly one executor call. Do not add planner, router, specialist,
verification, or provider calls. The shared policy executor owns model execution.

## Streaming tools

All streaming tools use hosted NVIDIA video through the shared runtime. Keep generated
tools declarative and put event/scene behavior entirely in `TOOL_PROMPT`.

### Hosted-video streaming tools

Use `EXECUTION_MODE = "hosted_video_streaming"` only when the requested result
requires recent history or comparison across time. Generate only `TOOL_NAME`,
that execution mode, explicit literal `TOOL_POLICY`, required literal
`VIDEO_CONFIG`, optional literal `OUTPUT_CONFIG`, and a task-specific
`TOOL_PROMPT`. The runtime reads and validates the same policy for take-photo
and independently scheduled streaming execution. The prompt must explain the
chronological, early/late, before/after, duration, sequence, or state-change
evidence the VLM should inspect. The shared
runtime supplies the rolling buffer, MP4 encoding, hosted request, event filtering,
deduplication, and output. Never implement RTSP/FFmpeg, frame buffering, async
loops, provider calls, card parsing, before/after state, or take-photo imports.

Temporal describes the **input context**: several ordered recent frames. It does
not imply that every temporal tool has before/after fields or persistent state.
Choose the output contract independently from the task:

- Prefer concise plain text for recognition tasks such as a recent sign-language
  gesture. Ask for only the final sentence or phrase and omit `OUTPUT_CONFIG`
  unless filtering settings are needed.
- Use a task-specific structured schema only when the runtime supports and the
  task genuinely needs structured fields. The prompt must name those fields and
  still yield a concise user-facing result through the runtime adapter.
- Use explicit state-change fields only when the requested output depends on
  comparing states.

Never copy an unrelated tool's output schema. In particular,
`OUTPUT_CONFIG = {"schema": "played_card_event"}` is exclusively for played-card
detection. Only that explicit declaration may request card JSON or fields such
as `before_cards`, `after_cards`, and `played_card`. Sign-language and other
generic temporal tools must not mention or declare card fields.

Although hosted-video declarations contain no `main` function, the shared Take
Photo runtime reads the same `TOOL_PROMPT` and performs one single-image call.

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
