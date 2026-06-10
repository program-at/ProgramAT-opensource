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
Before coding, include a Task Pipeline with the smallest number of stages needed. Create a new stage only when capability, modality, latency requirement, or output type changes. For each stage, specify one capability category: `object_detection`, `object_localization`, `OCR`, `visual_understanding`, `visual_reasoning`, `navigation`, `summarization`, or `general_reasoning`. Outputs from earlier stages should be reused by later stages. Use the model router for all model-backed operations; the router decides whether to use a specialized detector, OCR engine, VLM, LLM, or other backend. Specialized detectors are preferred for pure detection/localization/counting, but they are not automatically sufficient for fine-grained identification, attributes, license plates, visual comparison, or contextual reasoning. In those cases use `visual_understanding`, `visual_reasoning`, `OCR`, or a multi-stage pipeline. Generated tools should not hardcode Gemini, GPT, Claude, Llama, YOLO, Google Vision, or other provider/model names, should not define provider-specific `DEFAULT_MODEL` constants, and should not call `litellm.completion()` directly unless there is no router-compatible path.

**Alternatives Considered**
<!-- Describe any alternative solutions or features you've considered. -->

**Example usage**
<!-- Describe an example situation the tool would be used in and how it could work -->

**Custom GPT**
<!-- Should this tool, in live mode, leverage Gemini live and work basically as a custom GPT without the need to ask again?-->

**GPT Query**
<!-- If custom GPT, what is the query to be reasked every few seconds. Otherwise leave empty-->

**Additional Context**
<!-- Add any other context or screenshots about the feature request here. -->
Unless otherwise specified, in streaming mode, any verbal/text response should be limited to 15 words. No such limit applies to one-shot output.
