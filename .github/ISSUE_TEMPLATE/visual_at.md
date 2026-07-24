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

Copilot must implement the requested behavior in the generated tool file.
Declare `TOOL_NAME`, `EXECUTION_MODE = "take_photo"`, and lifecycle hooks such
as `on_take_photo`, `on_stream_start`, `on_frame`, and `on_stream_stop`.
Use `call_model()` directly with an explicit model name; default to
`gemini/gemini-3.1-flash-lite-preview` for ordinary visual tasks. The tool owns
frame selection, comparison, model orchestration, and output timing.

### Hosted-video streaming implementation guidance

For streaming, write `on_frame(runtime, frame)` and select any needed history
with `runtime.get_recent_frames(...)`. Implement task-specific similarity,
buffering, and temporal reasoning in the tool. Use `runtime.emit(...)` for one
or multiple results, and check cancellation or a tool-owned generation before
emitting stale work.

Tools belong in `tools/` and must be Python. Do not access API keys,
environment variables, arbitrary networks, subprocesses, filesystem writes,
the backend server, or WebSockets.
