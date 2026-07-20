# GPU-free hosted video streaming

The production experiment does not run local RTVI, RTSP, MediaMTX, Docker, or a
local GPU. It buffers recent ProgramAT camera JPEGs and sends input to exactly
one configured provider. Gemini receives four ordered 720x1280 JPEGs; NVIDIA
receives a short H.264 MP4 encoded with FFmpeg. Take-photo execution is unchanged.

The verified configuration is:

```env
STREAMING_EXECUTION_POLICY=hosted_video_only
VIDEO_VLM_PROVIDER=gemini
VIDEO_GEMINI_MODEL=gemini-3.1-flash-lite-preview
NVIDIA_VIDEO_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_VIDEO_API_KEY=<secret>
NVIDIA_VIDEO_MODEL=nvidia/nemotron-nano-12b-v2-vl
NVIDIA_VIDEO_INPUT_MODE=base64
HOSTED_VIDEO_WINDOW_SECONDS=6
HOSTED_VIDEO_INTERVAL_SECONDS=3
HOSTED_VIDEO_OVERLAP_SECONDS=3
HOSTED_VIDEO_MAX_TOKENS=256
HOSTED_VIDEO_OUTPUT_FPS=4
HOSTED_VIDEO_MAX_WIDTH=1280
HOSTED_VIDEO_JPEG_QUALITY=80
HOSTED_VIDEO_MAX_CLIP_BYTES=8388608
HOSTED_VIDEO_REQUEST_TIMEOUT_SECONDS=60
HOSTED_VIDEO_DUPLICATE_COOLDOWN_SECONDS=5
HOSTED_VIDEO_DEBUG_SAVE=false
```

Set `VIDEO_VLM_PROVIDER=nvidia` to use `NVIDIA_VIDEO_MODEL`. There is no
provider cascade: both providers share capture, buffering, parsing, and
WebSocket delivery, but only the selected provider is called. Gemini selects
the first, two uniformly spaced middle, and last real frame, then saves the
exact request as `backend/debug/last_hosted_images/frame-00.jpg` through
`frame-03.jpg` plus `metadata.json`. It does not invoke FFmpeg or create an MP4.

On 2026-07-20, `nvidia/nemotron-nano-12b-v2-vl` accepted a 29,215-byte H.264
MP4 through `/v1/chat/completions` using this content item:

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}
```

The hosted service does not expose an assumed file-upload API in this
NVIDIA integration. Consequently, NVIDIA base64 input is explicit and
size-bounded. Gemini sends the four ordered JPEG parts directly.

Tools declare `TOOL_NAME`, `EXECUTION_MODE = "hosted_video_streaming"`,
`TOOL_PROMPT`, and optional literal `VIDEO_CONFIG`/`OUTPUT_CONFIG`. The runtime
loads these through AST literal evaluation.

Verify the endpoint independently:

```bash
cd backend
./.venv/bin/python scripts/test_nvidia_hosted_video.py
```

Enable `HOSTED_VIDEO_DEBUG_SAVE=true` to retain each exact uploaded `clip.mp4`,
its unique source JPEGs, and `metadata.json` under `backend/hosted_video_debug`.
Replay a saved played-card clip with:

```bash
cd backend
./.venv/bin/python scripts/test_nvidia_hosted_video.py \
  hosted_video_debug/<clip-directory>/clip.mp4 --played-card
```

Compare that same clip across NVIDIA models without changing production:

```bash
./.venv/bin/python scripts/compare_nvidia_video_models.py \
  hosted_video_debug/<clip-directory>/clip.mp4 \
  --expected-card "jack of diamonds"
```

This includes the production model, Qwen3-VL-30B-A3B-Instruct when the endpoint
returns it, advertised video models, and IDs in
`NVIDIA_VIDEO_COMPARISON_MODELS`. It prints raw output, parsed cards,
validation/correctness, and latency for each model.

For app testing, restart the backend, select `played_card_rtvi`, start streaming,
keep the cards visible for at least five seconds, play one card, wait for hosted
inference, then stop streaming. With Gemini, expected logs show the four source
indices and timestamps, JPEG preprocessing and byte size, image-token usage,
inference latency, parsed output, final message, and idempotent cleanup.

For NVIDIA only, the FFmpeg command uses `-framerate 4 -c:v libx264 -preset veryfast -pix_fmt
yuv420p -movflags +faststart` with width bounded to 1280. Latency is approximately
the five-second collection interval plus CPU encoding and hosted inference.
