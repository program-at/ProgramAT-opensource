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
Assume each generated tool implements one user-facing task. If this issue enumerates multiple stages, implement those stages sequentially inside the generated tool and pass intermediate outputs from earlier stages into later stages. Each stage should include stage name, goal, capability, expected output, and input when it consumes an earlier stage. For any issue with a `Task Stages` section: each model-backed stage MUST map to its own `copilot_llm_call(...)`; do not merge multiple stage capabilities into one call; and use exactly the stage capability as `task_category`. For any LLM or VLM operation, generated tools must call the existing Copilot-routed backend interface through the existing client with `from model_router_client import copilot_llm_call`. Choose only from these task categories for each `copilot_llm_call(task_category=...)`: `general_reasoning`, `ocr`, `object_detection`, `map_web`, `spatial_relationship`, `navigation`, `camera_motion`, or `video`. Never use `visual_reasoning`. The model router only selects the most appropriate model for a capability. Do not ask it to execute stages, manage workflows, pass outputs between stages, or orchestrate pipelines. Do not implement detection, OCR, VLM, LLM, model loading, or provider calls inside generated tool files. Generated tools must not implement model routing, create routers, create capability registries, create detector/OCR/LLM wrappers, create new model-router clients, choose provider/model names, resolve API keys, define provider-specific `DEFAULT_MODEL` constants, define `COCO_CLASSES`, search for model files, load `.pt` files, import provider SDKs, import detector libraries, call `YOLO(...)`, or call `litellm.completion()` directly.

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
