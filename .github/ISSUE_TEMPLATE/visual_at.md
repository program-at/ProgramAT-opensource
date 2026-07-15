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

For take-photo tools, follow the generated `Task Stages` exactly. Implement one
ordered `copilot_llm_call(...)` per stage using its declared capability and goal.
Pass the original image to each visual stage. Pass useful earlier `artifact`
values explicitly through `metadata={"previous_stage_artifact": ...}` and/or the
next stage's messages. Return only the final stage's `response`. Do not fuse the
stages into one prompt, select models, add retries, or add evaluators.

Tools belong in `tools/`, must be Python, and must expose `main(image,
input_data)`. Return concise audio-friendly text. Do not connect to the backend
server or use WebSockets; the backend supplies the image and delivers the result.
