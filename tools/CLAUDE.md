# tools/ — ProgramAT vision tools

Each Python file is one assistive tool. The backend executes it with a camera
image and speaks the returned value. Tools do not use WebSockets or import one
another.

Expose `main(image, input_data)` with exactly two parameters. `image` is an
OpenCV BGR array and `input_data` is a dictionary. Return concise,
audio-friendly text; do not print.

For a take-photo tool, use this shape:

```python
from litellm_utils import call_take_photo_vlm

TOOL_NAME = "tool_name"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "One concise task-specific fused prompt."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
```

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
