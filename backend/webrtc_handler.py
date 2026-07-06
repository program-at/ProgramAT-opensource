"""
WebRTC handler for video+audio ingestion.

Exposes a single HTTP POST /offer endpoint that completes WebRTC signaling.
Once negotiated, incoming video frames are buffered and sent to Gemini as a
short clip at creation time.  Audio RMS is logged but not yet acted on —
wire `on_audio_rms` to whatever downstream needs it.

This module is completely independent of the WebSocket frame pipeline used
by tools.  Tools continue to receive PIL.Image via run_streaming_tools.

Usage (in stream_server.py main()):
    from webrtc_handler import webrtc_offer_handler, cleanup_webrtc_peers
    app.router.add_post('/offer', webrtc_offer_handler)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from typing import Callable, Awaitable

import numpy as np
from aiohttp import web
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional aiortc import — server starts normally even without it installed
# ---------------------------------------------------------------------------
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger.warning(
        "aiortc not installed — WebRTC /offer endpoint will return 503. "
        "Run: uv add aiortc av"
    )

# ---------------------------------------------------------------------------
# Active peer connections (keep refs so GC doesn't collect them)
# ---------------------------------------------------------------------------
_pcs: set["RTCPeerConnection"] = set()  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Frame buffer: collects PIL frames from a WebRTC video track
# ---------------------------------------------------------------------------
class _FrameBuffer:
    """Accumulates PIL frames from one WebRTC video track."""

    def __init__(self, max_frames: int = 150, max_seconds: float = 5.0):
        self._frames: list[Image.Image] = []
        self._max_frames = max_frames
        self._max_seconds = max_seconds
        self._start: float = time.monotonic()
        self._done = asyncio.Event()

    def add(self, pil_image: Image.Image) -> None:
        if self._done.is_set():
            return
        self._frames.append(pil_image)
        elapsed = time.monotonic() - self._start
        if len(self._frames) >= self._max_frames or elapsed >= self._max_seconds:
            self._done.set()

    async def wait_complete(self) -> list[Image.Image]:
        await self._done.wait()
        return self._frames

    def frames(self) -> list[Image.Image]:
        return self._frames


# ---------------------------------------------------------------------------
# Gemini clip submission
# ---------------------------------------------------------------------------
async def _submit_clip_to_gemini(
    frames: list[Image.Image],
    prompt: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
) -> str:
    """
    Encode frames as an in-memory MJPEG-style sequence and send to Gemini.

    For short clips (< ~20 MB encoded) uses the inline bytes approach.
    Returns the model's text response.
    """
    if not frames:
        return "No frames captured."

    try:
        import google.generativeai as genai
        import tempfile, pathlib

        genai.configure(api_key=api_key)

        # Build a simple MP4 from frames using av
        import av as _av  # type: ignore

        buf = io.BytesIO()
        with _av.open(buf, mode="w", format="mp4") as container:
            stream = container.add_stream("h264", rate=10)
            stream.width = frames[0].width
            stream.height = frames[0].height
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                av_frame = _av.VideoFrame.from_image(frame)
                av_frame = av_frame.reformat(format="yuv420p")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            for packet in stream.encode(None):  # flush
                container.mux(packet)

        video_bytes = buf.getvalue()

        # Upload via File API for reliability
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        uploaded = genai.upload_file(tmp_path, mime_type="video/mp4")
        pathlib.Path(tmp_path).unlink(missing_ok=True)

        # Wait for processing
        import time as _time
        for _ in range(20):
            if uploaded.state.name == "ACTIVE":
                break
            _time.sleep(1)
            uploaded = genai.get_file(uploaded.name)

        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content([uploaded, prompt])
        return response.text

    except Exception as exc:
        logger.error(f"[WebRTC] Gemini clip submission failed: {exc}", exc_info=True)
        return f"Gemini submission failed: {exc}"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _compute_rms(av_frame: object) -> float:
    """Return normalised RMS for one audio frame."""
    try:
        import av as _av  # type: ignore
        frame = av_frame  # type: _av.AudioFrame
        arr = frame.to_ndarray().flatten().astype(np.float32)
        if arr.size == 0:
            return 0.0
        # int16 PCM → normalise by 32768; float formats are already [-1,1]
        if arr.max() > 1.0:
            arr /= 32768.0
        return float(np.sqrt((arr ** 2).mean()))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------
async def webrtc_offer_handler(request: web.Request) -> web.Response:
    """
    POST /offer

    Body JSON:
        sdp         : SDP offer string
        type        : "offer"
        client_id   : (optional) identifier for logging
        prompt      : (optional) Gemini prompt to run on the captured clip
        clip_seconds: (optional) how many seconds of video to collect (default 5)

    Returns JSON with { sdp, type } SDP answer.
    After signaling, the server collects `clip_seconds` of video, builds an
    MP4, and submits it to Gemini with `prompt`.  The result is logged; wire
    on_clip_result callback below to route it back to your app.
    """
    if not AIORTC_AVAILABLE:
        return web.Response(status=503, text="aiortc not installed on server")

    try:
        params = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    client_id = params.get("client_id", request.remote or "webrtc")
    prompt = params.get("prompt", "Describe what you see and hear in this video clip.")
    clip_seconds = float(params.get("clip_seconds", 5.0))
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_model = os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))

    pc = RTCPeerConnection()
    _pcs.add(pc)
    logger.info(f"[WebRTC] New peer connection for client={client_id}")

    buf = _FrameBuffer(max_frames=int(clip_seconds * 30), max_seconds=clip_seconds)

    @pc.on("connectionstatechange")
    async def on_state() -> None:
        logger.info(f"[WebRTC] {client_id} connection state → {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            _pcs.discard(pc)
            buf._done.set()  # unblock any waiting coroutine

    @pc.on("track")
    def on_track(track: object) -> None:
        logger.info(f"[WebRTC] {client_id} received track kind={track.kind}")  # type: ignore[attr-defined]

        if track.kind == "video":  # type: ignore[attr-defined]
            async def run_video() -> None:
                try:
                    while not buf._done.is_set():
                        frame = await track.recv()  # type: ignore[attr-defined]
                        pil = frame.to_image()
                        buf.add(pil)
                except Exception as exc:
                    logger.debug(f"[WebRTC] video track ended for {client_id}: {exc}")
                finally:
                    buf._done.set()

            asyncio.ensure_future(run_video())

        elif track.kind == "audio":  # type: ignore[attr-defined]
            async def run_audio() -> None:
                try:
                    while True:
                        frame = await track.recv()  # type: ignore[attr-defined]
                        rms = _compute_rms(frame)
                        logger.debug(f"[WebRTC] audio RMS={rms:.4f} client={client_id}")
                        # Hook: extend here for speech-to-text or audio events
                except Exception:
                    pass

            asyncio.ensure_future(run_audio())

    # Complete SDP negotiation
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Schedule Gemini submission after clip is collected (non-blocking)
    async def _submit_after_capture() -> None:
        frames = await buf.wait_complete()
        logger.info(f"[WebRTC] clip collected: {len(frames)} frames for {client_id}")
        if frames and gemini_api_key:
            result = await _submit_clip_to_gemini(
                frames, prompt, gemini_api_key, model=gemini_model
            )
            logger.info(f"[WebRTC] Gemini result for {client_id}: {result[:200]}")
            # TODO: route result back to the originating WebSocket client by
            #       client_id if real-time delivery is needed.
        else:
            logger.warning(f"[WebRTC] no frames or no API key for {client_id}")

    asyncio.ensure_future(_submit_after_capture())

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


async def cleanup_webrtc_peers() -> None:
    """Close all open peer connections cleanly (call on server shutdown)."""
    for pc in list(_pcs):
        await pc.close()
    _pcs.clear()
