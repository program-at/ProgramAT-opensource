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

<!-- Enter exactly: take-photo or streaming. -->

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

Every take-photo tool must define exactly one `TOOL_PROMPT`, call
`call_take_photo_baseline_vlm` exactly once, and return its answer directly.

Tools belong in `tools/`, must be Python, and must expose `main(image,
input_data)`. Return concise audio-friendly text. Do not connect to the backend
server or use WebSockets; the backend supplies the image and delivers the result.
