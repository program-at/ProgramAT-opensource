---
name: Visual Assistive Technology
about: Propose a new visual assistive tool
title: ''
labels: enhancement
assignees: ''
---
<!-- Template: VAT -->
<!-- ORIGINAL_PROMPTS
-->

## Tool name

<!-- A short Python-friendly name for the tool. -->

## Task

<!-- What should the tool determine from the camera image? -->

## Expected output

<!-- What should the spoken answer contain? -->

## Constraints / examples

<!-- Important constraints, edge cases, and example inputs or answers. -->

## Mode

<!-- Enter exactly: take_photo or hosted_video_streaming. -->
<!-- This is a parser suggestion. Copilot must correct it if the preserved
original request clearly requires a different visual context. -->

<!-- Use take_photo for a static current-frame task, even if it may be run
continuously. Use hosted_video_streaming only when the answer requires recent
history, sequence, duration, before/after comparison, or what just happened. -->

### Take-photo implementation guidance

For a take-photo tool, author one concise, task-specific `TOOL_PROMPT` from
Task, Expected output, and Constraints / examples. Preserve the requested
behavior and output format, make the final answer accessible and audio-friendly,
and include a clear fallback when the requested visual information is unavailable.
Before writing the prompt, determine whether the task can be completed as one
operation or contains genuinely dependent subproblems. Default to no steps. If
one operation is sufficient, use one direct instruction even when the issue has
multiple output constraints. Do not create steps merely to restate the issue
fields. Only when a later conclusion depends on earlier visual findings should
the single fused prompt contain a concise ordered sequence; numbered instructions
are optional when they improve reliability.

The sequence must request only the final answer. Never turn prompt-level steps
into runtime stages, multiple model calls, router stages, specialist calls,
cascades, evaluators, or verification calls. Do not include capability names
unless naturally needed.

Every take-photo tool must define exactly one `TOOL_PROMPT`, emit a literal
`TOOL_POLICY`, call `execute_tool_policy` exactly once with that policy, and
return its answer directly. The ordinary default is
`{"strategy": "single", "models": ["gemini-3.1-flash-lite"]}`.
It must declare `EXECUTION_MODE = "take_photo"` and must not declare
`VIDEO_CONFIG`. The same tool may be streamed as independent single-frame calls.

### Hosted-video streaming implementation guidance

For `hosted_video_streaming`, generate only `TOOL_NAME`, `EXECUTION_MODE =
"hosted_video_streaming"`, explicit literal `TOOL_POLICY`, required literal
`VIDEO_CONFIG`, optional literal `OUTPUT_CONFIG`, and a task-specific
`TOOL_PROMPT`. The shared runtime owns FFmpeg clip encoding,
hosted NVIDIA requests,
filtering, deduplication, result delivery, and cleanup. Do not generate frame
processing, buffers, asynchronous loops, model calls, or take-photo imports.
`VIDEO_CONFIG` must include valid `window_seconds`, `interval_seconds`,
`minimum_span_seconds`, and `minimum_unique_frames`. `TOOL_PROMPT` must state
what chronological or early/late state-change evidence to compare.
Temporal input does not imply a before/after output schema. Prefer concise plain
text for recognition tasks. Declare `played_card_event` only for played-card
detection; never reuse its card fields for unrelated temporal tools.
The same temporal `TOOL_PROMPT` is also used by Take Photo with one current
image. It must use static evidence when available, never infer unseen motion,
and return a task-specific uncertainty response when one frame is insufficient.

Tools belong in `tools/` and must be Python. Static tools expose
`main(image, input_data)`; temporal hosted-video tools contain only the
declarative constants above. The shared runtime still supports Take Photo for
both contracts. Do not connect to the backend server or use WebSockets.
