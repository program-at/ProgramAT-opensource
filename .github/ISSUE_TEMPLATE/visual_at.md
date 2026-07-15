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

## Prompt strategy
<!-- Backend-selected take-photo prompt strategy. -->

## P1 exact prompt
<!-- Mechanically assembled P1 prompt; empty for Copilot P2. -->

For `no_planner`, copy the P1 exact prompt verbatim into `TOOL_PROMPT`.

For `copilot_fused_prompt`, author one concise, task-specific `TOOL_PROMPT` from
Task, Expected output, and Constraints / examples. Preserve the requested
behavior and output format, make the final answer accessible and audio-friendly,
and include a clear fallback when the requested visual information is unavailable.
Prefer the shortest prompt that captures the task. Do not force simple recognition,
OCR, classification, or identification tasks into steps. Use an ordered logical
sequence only when later reasoning genuinely depends on earlier visual findings.
The sequence remains inside one prompt and must request only the final answer.
Do not create runtime stages, capability metadata, routing, cascades, evaluators,
or specialist-model calls. Do not include capability names unless naturally needed.

Every take-photo tool must define exactly one `TOOL_PROMPT`, call
`call_take_photo_baseline_vlm` exactly once, and return its answer directly.

Tools belong in `tools/`, must be Python, and must expose `main(image,
input_data)`. Return concise audio-friendly text. Do not connect to the backend
server or use WebSockets; the backend supplies the image and delivers the result.
