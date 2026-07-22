# tools/ — ProgramAT vision tools

Each Python file is one assistive tool. The backend executes it with a camera
image and speaks the returned value. Tools do not use WebSockets or import one
another.

Expose `main(image, input_data)` with exactly two parameters. `image` is an
OpenCV BGR array and `input_data` is a dictionary. Return concise,
audio-friendly text; do not print.

For a take-photo tool, use this shape:

```python
from model_execution import execute_tool_policy

TOOL_NAME = "tool_name"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "One concise task-specific fused prompt."
TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return execute_tool_policy(
        image=image, prompt=TOOL_PROMPT, policy=TOOL_POLICY, tool_name=TOOL_NAME
    )
```

Tools must normally declare a literal `TOOL_POLICY` using the schema in
`backend/strategy_registry.py` and pass it as `policy=TOOL_POLICY`. The runtime
uses one `gemini-3.1-flash-lite` call only as a compatibility fallback for a
missing or invalid policy. Do not implement custom model orchestration.

Make exactly one helper call and return it directly. Do not add another model or
specialist call, verification pass, fallback model, model name, or provider SDK.
Author one concise fused prompt following the detailed guidance in
`.github/copilot-instructions.md`.

Static `take_photo` tools may also stream; each selected frame is processed
independently through the same one-image prompt. Static tools must not declare
`VIDEO_CONFIG` or keep temporal state.

For `hosted_video_streaming`, declare `TOOL_NAME`, `EXECUTION_MODE`,
`TOOL_PROMPT`, required literal `VIDEO_CONFIG`, and optional literal
`OUTPUT_CONFIG` only. The prompt must describe the chronological evidence to
compare. The shared runtime owns the continuous video
session and output filtering. Do not import take-photo helpers or implement
buffering, FFmpeg, model calls, asynchronous loops, or before/after state.

Temporal input and output schema are separate decisions. Generic temporal
recognition tools should request one concise plain-text result. Declare
`OUTPUT_CONFIG = {"schema": "played_card_event"}` only for played-card detection;
never use card fields for sign language or another unrelated task.

Take Photo works for temporal tools through the shared runtime and supplies one
current image to the same `TOOL_PROMPT`. Write temporal prompts to use ordered
motion evidence when multiple frames exist, use only visible static evidence
with one frame, and return a task-specific uncertainty response rather than
inventing unseen motion.
