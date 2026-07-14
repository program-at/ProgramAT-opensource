# NVIDIA streaming experiments

ProgramAT has two independent NVIDIA experiments. Select exactly one with
`NVIDIA_STREAMING_MODE`; neither mode falls back to the other or to ProgramAT's
normal inference pipeline after an error.

Accepted values are:

- `disabled` or `original`: existing ProgramAT streaming, unchanged.
- `rtvi`: RTSP plus the separately deployed NVIDIA RTVI microservice.
- `hosted_multiframe`: recent frames sent directly to NVIDIA's hosted VLM API.

## hosted_multiframe

```text
ProgramAT JPEG frames
    -> bounded rolling window
    -> uniformly sampled chronological frames
    -> NVIDIA hosted VLM API
    -> tool_stream_result
```

This mode requires an NVIDIA hosted API key, but does not require a local GPU,
RTVI deployment, MediaMTX, RTSP, Kafka, or Redis. It bypasses ProgramAT's planner,
step decomposition, model/capability router, cascade, evaluator, CLIP scheduler,
debounce/stability logic, Gemini streaming, and per-frame VLM calls.

Configure `backend/.env`:

```dotenv
NVIDIA_STREAMING_MODE=hosted_multiframe
NVIDIA_HOSTED_API_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_HOSTED_API_KEY=<your-key>
NVIDIA_HOSTED_MODEL=
NVIDIA_HOSTED_WINDOW_SECONDS=2
NVIDIA_HOSTED_SAMPLE_FRAMES=6
NVIDIA_HOSTED_REQUEST_INTERVAL_SECONDS=1.5
NVIDIA_HOSTED_MAX_IN_FLIGHT=1
NVIDIA_HOSTED_REQUEST_TIMEOUT_SECONDS=60
NVIDIA_HOSTED_MAX_TOKENS=80
```

When the model is empty, ProgramAT calls `GET /models`, logs all returned model
IDs, and selects the first entry positively identified as vision-capable. An
explicit model must appear in that response and advertise vision support.

NVIDIA's hosted Chat Completions schema supports `image_url`; specific models
also support `video_url`. ProgramAT uses native base64 H.264 MP4 only when model
metadata or NVIDIA's model-specific documentation positively identifies video
support. Otherwise, it sends one request containing text followed by ordered,
base64 JPEG `image_url` items. It does not retry in a different format if the
selected model rejects the request.

The rolling buffer is bounded by both time and count. Each request interval
uniformly samples up to the configured frame count in chronological order. If a
request is still running at the next interval, that interval is skipped rather
than queued.

The server advertises a hosted-only capture interval derived from the configured
window and sample count. The app sends every captured frame in this mode so the
default two-second/six-frame window can actually be populated. Original and RTVI
modes retain the app's existing every-third-frame behavior.

### Standalone hosted API validation

Run this before enabling streaming, using one or more local JPEG/PNG files:

```bash
backend/.venv/bin/python backend/test_nvidia_hosted_api.py \
  backend/received_frames/frame-one.jpg \
  backend/received_frames/frame-two.jpg
```

The script prints all models, validates the selected model, sends a single-image
request, then sends one ordered multi-image request. If multiple images are
rejected, it prints the complete HTTP error body and does not silently change
the request format.

This experiment benchmarks NVIDIA-hosted VLM quality, temporal reasoning over
recent frames, and latency under ProgramAT's hosted multiframe interval loop. It
does not benchmark DeepStream, RTSP, the RTVI scheduler, or RTVI chunking.

Official capability references:

- [NVIDIA Nemotron Nano 12B V2 VL model card](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl/modelcard)
- [NVIDIA hosted Nemotron Omni API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-infer)
- [NVIDIA VLM image/video request format](https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/qwen/api.html)

## rtvi

```text
ProgramAT JPEG frames
    -> FFmpeg
    -> MediaMTX RTSP
    -> NVIDIA RTVI microservice
    -> SSE captions
    -> tool_stream_result
```

This retains the previously implemented infrastructure experiment. It requires
FFmpeg, an RTSP server reachable by both ProgramAT and RTVI, and a separately
deployed RTVI service.

Start the included MediaMTX service:

```bash
docker compose -f docker-compose.rtvi.yml up -d
```

Configure:

```dotenv
NVIDIA_STREAMING_MODE=rtvi
NVIDIA_RTVI_BASE_URL=http://localhost:8000
NVIDIA_RTVI_MODEL=
NVIDIA_RTVI_CHUNK_DURATION_SECONDS=2
NVIDIA_RTVI_CHUNK_OVERLAP_SECONDS=0
NVIDIA_RTVI_REQUEST_TIMEOUT_SECONDS=30
PROGRAMAT_RTSP_PUBLIC_BASE_URL=rtsp://192.0.2.10:8554/programat
PROGRAMAT_RTSP_FPS=5
PROGRAMAT_RTSP_WIDTH=
PROGRAMAT_RTSP_HEIGHT=
```

`PROGRAMAT_RTSP_PUBLIC_BASE_URL` must be reachable by both FFmpeg and the RTVI
deployment. `localhost` inside an RTVI container is not the MediaMTX host.

Verify the service and a published stream:

```bash
curl --fail http://localhost:8000/v1/ready
curl --fail http://localhost:8000/v1/models
ffprobe -rtsp_transport tcp rtsp://192.0.2.10:8554/programat/<session-id>
```

## Returning to original streaming

Set either of the following and restart the backend:

```dotenv
NVIDIA_STREAMING_MODE=original
```

or:

```dotenv
NVIDIA_STREAMING_MODE=disabled
```

Both preserve the existing ProgramAT CLIP/router/cascade streaming behavior.
