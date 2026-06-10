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
Before coding, include a Capability Pipeline with the smallest number of stages needed. Create a new stage only when capability, modality, latency requirement, or output type changes. For each stage, specify one capability category: `object_detection`, `object_localization`, `OCR`, `visual_understanding`, `visual_reasoning`, `navigation`, `summarization`, `general_reasoning`, or `simple_parsing`. Outputs from earlier stages should be reused by later stages. Capability categories are declarations, not implementation requirements. Generated tools should call backend-provided capability interfaces and trust the existing backend capability layer and centralized model router. The backend decides model selection, fallback logic, provider selection, detector selection, and OCR selection. Generated tools must not implement model routing, create routers, create capability registries, create detector/OCR/LLM wrappers, choose provider/model names, define provider-specific `DEFAULT_MODEL` constants, or call `litellm.completion()` directly unless there is no backend capability path.

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
