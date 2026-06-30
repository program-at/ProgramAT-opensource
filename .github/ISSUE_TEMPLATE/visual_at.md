---
name: Visual Assistive Technology
about: Propose a new mode of visual assistive technology
title: ''
labels: enhancement
assignees: ''

---
<!-- Template: VAT -->
<!-- ORIGINAL_PROMPTS
-->

**Feature Description**
<!-- A clear and concise description of the tool you'd like. -->

**Problem It Solves**
<!-- Describe the problem this tool would solve. -->

**Proposed Solution**
<!-- Describe how you envision this tool working. -->

**Implementation details**
<!-- Any particular models or libraries that should be employed -->
Assume each generated tool implements one user-facing task. If this issue enumerates multiple stages, execute one ordered `copilot_llm_call(...)` per stage and explicitly pass useful structured artifacts to later calls with `metadata={"previous_stage_artifact": ...}`. Use the stage capability as `capability`. Choose only from these capabilities: `general_reasoning`, `ocr`, `object_detection_localization`, `structured_visual_understanding`, `spatial_reasoning`, `navigation`, `camera_motion`, or `temporal_reasoning`. Never use `visual_reasoning`. The backend may evaluate and escalate reasoning capabilities according to the execution policy. Generated tools must not choose implementations, models, providers, detector backends, fallback order, retries, or verification logic. Do not implement detection, OCR, VLM, LLM, model loading, or provider calls inside generated tool files. Generated tools must not create routers, capability registries, detector/OCR/LLM wrappers, new model-router clients, provider-specific `DEFAULT_MODEL` constants, `COCO_CLASSES`, `.pt` model loading/discovery logic, or direct provider calls.

**Alternatives Considered**
<!-- Describe any alternative solutions or features you've considered. -->

**Example usage**
<!-- Describe an example situation the tool would be used in and how it could work -->

**Live Mode**
<!-- Should this tool, in live mode, use the backend-managed live multimodal mode without the need to ask again?-->

**Live Query**
<!-- If live mode is enabled, what is the query to be reasked every few seconds. Otherwise leave empty-->

**Additional Context**
<!-- Add any other context or screenshots about the feature request here. -->
Unless otherwise specified, in streaming mode, any verbal/text response should be limited to 15 words. No such limit applies to one-shot output.
