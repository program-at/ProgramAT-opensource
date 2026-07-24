# ProgramAT Copilot instructions

ProgramAT tools are executable Python modules in `tools/`. Generated code owns
the actual assistive behavior: model selection, prompts, call timing, frame
selection, temporal comparison, concurrency, progressive output, and
tool-specific state. The backend only supplies frames, isolated state,
cancellation, credentials inside approved model functions, and WebSocket
transport.

## Public execution contract

Declare:

```python
TOOL_NAME = "concise_snake_case_name"
EXECUTION_MODE = "take_photo"
```

Implement whichever lifecycle hooks the tool needs:

```python
async def on_take_photo(runtime, image, input_data): ...
async def on_stream_start(runtime, input_data): ...
async def on_frame(runtime, frame): ...
async def on_stream_stop(runtime): ...
```

Take Photo calls `on_take_photo`. Streaming calls `on_stream_start` once,
`on_frame` for collected frames, and `on_stream_stop` during cleanup. These
functions and every local helper in the tool file are actually executed.

`frame` provides `frame_id`, `timestamp`, `image`, `width`, `height`, and
`image_base64`. Available infrastructure is:

```python
runtime.current_request_id
runtime.current_frame
runtime.get_latest_frame()
runtime.get_recent_frames(count=None, seconds=None)
runtime.get_state(key, default=None)
runtime.set_state(key, value)
runtime.clear_state()
runtime.is_cancelled()
await runtime.emit(text, partial=False, final=False, replace=False, metadata=None)
```

State is isolated by client, tool source version, and streaming session. Keep
scene anchors, prior outputs, analysis generations, and temporary tasks in
runtime state rather than module globals.

## Model calls

For most models:

```python
from litellm_utils import call_model, extract_text

response = await asyncio.to_thread(
    call_model,
    "gemini/gemini-3.1-flash-lite-preview",
    [{"role": "user", "content": prompt}],
    [image],
    {"timeout": 60},
)
text = extract_text(response)
```

Use `call_openai_responses_model()` only when the selected model requires the
OpenAI Responses API. Inspect `backend/model_registry.py` before selecting a
model. The registry is advisory; the backend will not replace the model written
in the tool. Default to Gemini 3.1 Flash Lite for ordinary general visual work.

Tools may explicitly call multiple models, branch on prior results, retry when
appropriate, or use `asyncio.create_task`, `gather`, or `as_completed`.

## Examples

### One-frame, one-model Take Photo

```python
async def on_take_photo(runtime, image, input_data):
    del runtime, input_data
    response = await asyncio.to_thread(
        call_model, "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}], [image],
    )
    return extract_text(response)
```

### Tool-written streaming similarity

```python
def similarity(a, b):
    return float(np.dot(a, b))

async def on_frame(runtime, frame):
    current = await asyncio.to_thread(embed_frame, frame.image)
    anchor = runtime.get_state("anchor")
    if anchor is not None and similarity(anchor, current) >= 0.985:
        return
    runtime.set_state("anchor", current)
    # The tool decides what to call and emit here.
```

### Several frames from recent history

```python
async def on_frame(runtime, frame):
    recent = runtime.get_recent_frames(count=8, seconds=4)
    selected = recent[::2]
    if len(selected) < 2:
        return
    response = await asyncio.to_thread(
        call_model, MODEL, [{"role": "user", "content": TEMPORAL_PROMPT}],
        [item.image for item in selected],
    )
```

### Progressive parallel output

```python
async def on_frame(runtime, frame):
    generation = runtime.get_state("generation", 0) + 1
    runtime.set_state("generation", generation)

    async def run(model):
        response = await asyncio.to_thread(
            call_model, model, [{"role": "user", "content": TOOL_PROMPT}],
            [frame.image],
        )
        if not runtime.is_cancelled() and runtime.get_state("generation") == generation:
            await runtime.emit(extract_text(response), partial=True,
                               metadata={"model": model})

    await asyncio.gather(*(run(model) for model in MODELS), return_exceptions=True)
    if not runtime.is_cancelled() and runtime.get_state("generation") == generation:
        await runtime.emit("", final=True)
```

### Different Take Photo and Streaming behavior

```python
async def on_take_photo(runtime, image, input_data):
    return await analyze_once(image)

async def on_stream_start(runtime, input_data):
    runtime.set_state("last_processed_id", 0)

async def on_frame(runtime, frame):
    if frame.frame_id - runtime.get_state("last_processed_id", 0) < 5:
        return
    runtime.set_state("last_processed_id", frame.frame_id)
    await runtime.emit(await analyze_once(frame.image), final=True)
```

## Output and accessibility

Emit concise, speech-ready results for blind and low-vision users. Say when
visual evidence is insufficient. Navigation guidance must use body-relative
directions or stable nonvisual/structural cues, never color or another
visual-only landmark as the sole cue.

The tool decides whether output appends, replaces, is partial, or is final. Do
not access the WebSocket directly.

## Safety and validation

Allowed:

- `litellm_utils.call_model`, `extract_text`, and
  `call_openai_responses_model`;
- OpenCV, NumPy, PIL, CLIP, and local helper functions;
- normal Python control flow and asyncio orchestration;
- runtime frame history, state, cancellation, and emission.

Forbidden:

- raw API keys or environment-variable access;
- direct provider SDKs or arbitrary networking;
- WebSockets and backend transport internals;
- subprocesses or unrestricted filesystem writes;
- other clients, sessions, or shared backend globals.

Use the simplest implementation satisfying the request. Old declarative
`TOOL_POLICY` tools are compatibility-only; new code must not rely on the
backend to choose models, compare frames, or orchestrate cascades.
