# GPU-free hosted NVIDIA video streaming

The production experiment does not run local RTVI, RTSP, MediaMTX, Docker, or a
local GPU. It buffers recent ProgramAT camera JPEGs, writes chronological frames
to a temporary directory, encodes a short H.264 MP4 with FFmpeg, and sends that
clip to NVIDIA's hosted OpenAI-compatible endpoint. Take-photo execution is
unchanged.

The verified configuration is:

```env
STREAMING_EXECUTION_POLICY=hosted_video_only
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

On 2026-07-20, `nvidia/nemotron-nano-12b-v2-vl` accepted a 29,215-byte H.264
MP4 through `/v1/chat/completions` using this content item:

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}
```

The hosted service does not expose an assumed file-upload API in this
integration. Consequently, base64 is explicit, size-bounded, and the only
supported input mode. There is no image or Gemini fallback.

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

For app testing, restart the backend, select `played_card_rtvi`, start streaming,
keep the cards visible for at least five seconds, play one card, wait for hosted
inference, then stop streaming. Expected logs show session start, a five-second
window, FFmpeg encoding, base64 preparation, inference latency, accepted or
suppressed output, and idempotent cleanup.

The FFmpeg command uses `-framerate 4 -c:v libx264 -preset veryfast -pix_fmt
yuv420p -movflags +faststart` with width bounded to 1280. Latency is approximately
the five-second collection interval plus CPU encoding and hosted inference.
