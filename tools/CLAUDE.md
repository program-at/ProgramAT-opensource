# tools/ — ProgramAT vision tools

Each `.py` file here is one assistive tool. The backend loads a tool's source,
`exec()`s it against a single camera frame, and speaks the result to a blind user.
Tools never import each other and never touch app/backend WebSocket layer — they
get an image, they return text. (`litellm_utils.py` is a shared helper, not a tool.) 
You should only touch the network if making a call to a model or API.

## The contract

A tool MUST define a function — `main` by convention (`run` and `process_image`
also work):

```python
def main(image, input_data=None):
    ...
    return result
```

- **Exactly two parameters.** The server checks the signature and silently skips a
  `main` with any other arity — the tool then does nothing.
- **`image`** is an OpenCV `numpy` array in **BGR** order, and may be `None`. Always
  guard it, and convert BGR→RGB before sending it to an LLM or PIL.
- **`input_data`** is always a `dict`. Read config with `.get()`; assume nothing.
- **Return** a plain `str` (spoken via TTS) or a `dict` with `text` and/or `audio`.
  `audio` is `{'type', 'text', 'rate', 'interrupt'}`, where `type` is one of
  `speech`, `beep_high`, `beep_low`, `success`, `warning`, `error`. Other dict keys
  are ignored. Return `""` to stay silent.

## Rules that prevent silent failure

- **Never `print()`.** The server captures stdout and reads it aloud to the user.
  Return information; don't print it.
- **Never raise.** Streaming-mode exceptions are swallowed with no feedback. Catch
  errors and return an error string or an `audio` dict with `type: 'error'`.
- **Cache heavy models** in the injected `yolo_model_cache` global —
  `globals().get('yolo_model_cache', {})` — not on every call. Reloading a model
  per frame makes streaming unusable. `door_detection.py` shows the pattern.
- **Streaming etiquette:** when run per frame, return `""` when nothing changed, or
  the user hears the same thing every ~500ms. Streaming output is capped at ~15 words.
- **Model-backed work:** treat capability categories as declarations, not
  implementation requirements. Assume each generated tool implements one
  user-facing task. If the issue enumerates multiple stages, execute one
  `copilot_llm_call()` per stage and explicitly pass useful artifact fields to
  the next call. Use the existing backend model router
  through the existing tool-facing client for capability execution. Do
  not implement routing, create routers, create capability registries, create
  detector/OCR/LLM wrappers, concrete detector functions, model-backed
  inference, model loading, or provider calls inside a generated tool.
  Do not choose candidate implementations inside the generated tool.
  Import `copilot_llm_call` from `model_router_client` for every stage.
  Canonical categories for new tools are `general_reasoning`, `ocr`,
  `object_detection_localization`, `structured_visual_understanding`,
  `spatial_reasoning`, `navigation`, `camera_motion`, and
  `temporal_reasoning`. Use only these category names. Include
  `metadata={"tool_name": "...", "route_text": "..."}` for routing logs.
  For `object_detection_localization`, also provide
  `metadata={"target_labels": ["requested label"]}`. Do not
  hardcode provider/model names, detector names, provider-specific
  `DEFAULT_MODEL` constants, `COCO_CLASSES`, direct `litellm.completion()`
  calls, `YOLO(...)` calls, detector library imports, provider SDK imports,
  `.pt` file loading, or model-file discovery logic. If the approved API cannot
  support the needed capability, add support to the centralized backend router
  instead of implementing it inside the tool.
- **Imports:** `import litellm_utils` bare (the server makes `tools/` the import
  root). Wrap optional dependencies in `try/except ImportError` — missing pip
  packages are auto-installed by the backend.
- Keep output short and audio-friendly: positional, body-relative language.

## Examples to copy from

- `object_recognition.py` — simple detection-style tool returning a plain string.
- `door_detection.py` — streaming, temporal smoothing, clock-face navigation.
- `scene_description.py`, `clothing_recognition.py` — routed vision-language tools.
- `appearance_check.py` — the most complete example: two-pass detect→verify,
  robust JSON parsing, fail-closed safety. Study it before writing an LLM tool.

## Adding a tool

1. Create `tools/your_tool.py` with `main(image, input_data)`.
2. Write a `backend/test_<tool>.py` script and verify with `python test_<tool>.py`.
3. Compose planner-produced stages explicitly with ordered `copilot_llm_call()`
   invocations. The tool owns artifact passing; the backend chooses the first
   configured implementation for each atomic capability call.
4. `MODEL_SETUP.md` covers model files; `.github/copilot-instructions.md` has the
   full tool-generation spec. `CONTRIBUTING.md` has the PR and review process.
