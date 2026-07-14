# tools/ — ProgramAT vision tools

Each Python file is one assistive tool. The backend executes it with a camera
image and speaks the returned value. Tools do not use WebSockets or import one
another.

Expose `main(image, input_data)` with exactly two parameters. `image` is an
OpenCV BGR array and `input_data` is a dictionary. Return concise,
audio-friendly text; do not print.

For a take-photo tool, use this shape:

```python
from litellm_utils import call_take_photo_baseline_vlm

TOOL_PROMPT = "A concise prompt describing this tool's visual task."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
```

Make exactly one helper call and return it directly. Do not add another model or
specialist call, verification pass, fallback, model name, or provider SDK.

For streaming tools, preserve existing streaming behavior and keep responses to
about 15 spoken words. Do not change NVIDIA hosted streaming or RTVI code.
