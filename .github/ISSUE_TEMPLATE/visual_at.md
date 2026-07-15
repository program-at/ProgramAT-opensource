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

## Fused VLM Prompt
<!-- Creation-time fused prompt; leave empty outside fused_prompt mode. -->

For a take-photo tool with a populated fused prompt, copy it verbatim into a
`FUSED_VLM_PROMPT` string constant and pass that constant to
`call_take_photo_baseline_vlm`. Do not invent, rewrite, or decompose the prompt.
If this section is empty, retain the existing `TOOL_PROMPT` behavior. The tool
must make no other model or specialist calls.

Tools belong in `tools/`, must be Python, and must expose `main(image,
input_data)`. Return concise audio-friendly text. Do not connect to the backend
server or use WebSockets; the backend supplies the image and delivers the result.
