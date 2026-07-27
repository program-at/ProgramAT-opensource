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

<!-- What should the tool determine from the camera image or recent frames? -->

## Expected output

<!-- What should the concise spoken answer contain? -->

## Constraints / examples

<!-- Important constraints, edge cases, and example inputs or answers. -->

## Streaming context

<!-- Describe whether Streaming can use a changed latest frame or needs recent chronological frames. -->

## Implementation guidance

Implement the requested behavior in one Python file in `tools/`. Normally support both Take Photo and Streaming with `on_take_photo`, `on_stream_start`, `on_frame`, and `on_stream_stop`. Define one concise, task-specific `TOOL_PROMPT` shared by both entry points. Default to one direct instruction; add ordered steps only when a later conclusion genuinely depends on earlier visual findings.

Choose models, frame selection, call timing, orchestration, and output behavior explicitly in the tool. Take Photo receives one current image. Streaming should decide whether to process a changed latest frame or selected chronological history through the runtime frame APIs. Use runtime state, cancellation, and `runtime.emit` as needed. Keep results concise, speech-ready, accessible to blind and low-vision users, and honest about unavailable or uncertain visual evidence.

Tools belong in `tools/` and use Python. Keep output concise and audio-friendly for blind and low-vision users, state uncertainty when evidence is insufficient, and restrict clock-face navigation to 9–12 and 1–3. Modify only the requested tool file unless the public runtime is genuinely insufficient.

Do not access credentials, environment variables, arbitrary networks, subprocesses, unrestricted filesystem writes, backend-private state, or WebSockets. Detailed APIs and examples are in `.github/copilot-instructions.md`.
