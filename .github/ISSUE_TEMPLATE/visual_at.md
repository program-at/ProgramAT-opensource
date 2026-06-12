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
Assume each generated tool implements one user-facing task. Do not plan subtasks or chain multiple model calls unless the user explicitly asks for that. For any LLM or VLM operation, generated tools must call the existing backend model router through the existing client with `from model_router_client import llm_call`. Choose the single nearest task category for `llm_call(task_category=...)`: `simple_parsing`, `object_detection`, `object_localization`, `ocr`, `visual_understanding`, `visual_reasoning`, `navigation`, `summarization`, `code_generation`, or `general_reasoning`. Do not implement detection, OCR, VLM, LLM, model loading, or provider calls inside generated tool files. Generated tools must not implement model routing, create routers, create capability registries, create detector/OCR/LLM wrappers, create new model-router clients, choose provider/model names, resolve API keys, define provider-specific `DEFAULT_MODEL` constants, define `COCO_CLASSES`, search for model files, load `.pt` files, import provider SDKs, import detector libraries, call `YOLO(...)`, or call `litellm.completion()` directly.

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
