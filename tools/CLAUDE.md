# tools/ — ProgramAT vision tools

Each Python file is one assistive tool. The backend executes it with a camera
image and speaks the returned value. Tools do not use WebSockets or import one
another.

Expose `main(image, input_data)` with exactly two parameters. `image` is an
OpenCV BGR array and `input_data` is a dictionary. Return concise,
audio-friendly text; do not print.

For a take-photo tool, import the established capability client:

```python
from model_router_client import copilot_llm_call
```

Follow the issue's `Task Stages` exactly: one ordered call per stage, with the
declared capability and step-specific goal. Pass useful prior artifacts to later
calls, keep the original image available to visual stages, and return the final
call's `response`. Do not fuse stages or choose models, retries, evaluators, or
fallbacks. See `.github/copilot-instructions.md` for the full handoff contract.

For streaming tools, preserve existing streaming behavior and keep responses to
about 15 spoken words. Do not change NVIDIA hosted streaming or RTVI code.
