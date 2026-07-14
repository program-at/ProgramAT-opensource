"""NVIDIA-hosted OpenAI-compatible VLM client and multiframe helpers."""

from __future__ import annotations

import base64
import io
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import aiohttp
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class NvidiaHostedError(RuntimeError):
    """Raised for hosted NVIDIA discovery, media, or inference failures."""


@dataclass(frozen=True)
class HostedModel:
    id: str
    supports_vision: bool
    supports_video: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class TimedFrame:
    timestamp: float
    image_data_uri: str


# These hosted models have explicit NVIDIA documentation for native video input.
# This is capability metadata only; it is never used as a default model choice.
DOCUMENTED_VIDEO_MODELS = {
    "nvidia/nemotron-nano-12b-v2-vl",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
}

VISION_ID_MARKERS = (
    "vision",
    "-vl",
    "_vl",
    "multimodal",
    "omni",
    "vila",
    "llava",
    "paligemma",
    "florence",
)


def _flatten_capability_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, str):
        values.add(value.lower())
    elif isinstance(value, dict):
        for key, nested in value.items():
            if nested not in (False, None, ""):
                values.add(str(key).lower())
                values.update(_flatten_capability_values(nested))
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            values.update(_flatten_capability_values(nested))
    return values


def parse_hosted_models(payload: Any) -> list[HostedModel]:
    """Parse `/models` and retain any advertised modality information."""
    if isinstance(payload, dict):
        entries = payload.get("data") or payload.get("models") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    models: list[HostedModel] = []
    for entry in entries:
        if isinstance(entry, str):
            raw = {"id": entry}
        elif isinstance(entry, dict):
            raw = entry
        else:
            continue
        model_id = str(raw.get("id") or raw.get("model") or "").strip()
        if not model_id:
            continue
        capability_values: set[str] = set()
        for key in (
            "capabilities",
            "modalities",
            "input_modalities",
            "supported_modalities",
            "model_type",
            "type",
        ):
            capability_values.update(_flatten_capability_values(raw.get(key)))
        id_lower = model_id.lower()
        supports_video = (
            model_id in DOCUMENTED_VIDEO_MODELS
            or "video" in capability_values
            or "video_url" in capability_values
        )
        supports_vision = (
            supports_video
            or bool({"image", "vision", "image_url"} & capability_values)
            or any(marker in id_lower for marker in VISION_ID_MARKERS)
        )
        models.append(HostedModel(model_id, supports_vision, supports_video, dict(raw)))
    return models


def select_hosted_model(models: Sequence[HostedModel], configured_model: str = "") -> HostedModel:
    if configured_model:
        selected = next((model for model in models if model.id == configured_model), None)
        if selected is None:
            raise NvidiaHostedError(
                f"Configured NVIDIA_HOSTED_MODEL '{configured_model}' was not returned by /models"
            )
        if not selected.supports_vision:
            raise NvidiaHostedError(
                f"Configured NVIDIA_HOSTED_MODEL '{configured_model}' is not advertised as vision-capable"
            )
        return selected
    selected = next((model for model in models if model.supports_vision), None)
    if selected is None:
        raise NvidiaHostedError("NVIDIA /models returned no vision-capable model")
    return selected


class RollingFrameBuffer:
    """Time-windowed, count-bounded, chronologically ordered frame storage."""

    def __init__(self, window_seconds: float, max_frames: int):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        self.window_seconds = window_seconds
        self._frames: deque[TimedFrame] = deque(maxlen=max_frames)

    def add(self, image_data_uri: str, timestamp: float) -> None:
        self._frames.append(TimedFrame(timestamp, image_data_uri))
        self.prune(timestamp)

    def prune(self, newest_timestamp: float) -> None:
        cutoff = newest_timestamp - self.window_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def snapshot(self, now: float | None = None) -> list[TimedFrame]:
        if now is not None:
            self.prune(now)
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


def uniformly_sample_frames(frames: Sequence[TimedFrame], sample_count: int) -> list[TimedFrame]:
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero")
    if len(frames) <= sample_count:
        return list(frames)
    if sample_count == 1:
        return [frames[-1]]
    last_index = len(frames) - 1
    indices = [round(index * last_index / (sample_count - 1)) for index in range(sample_count)]
    return [frames[index] for index in indices]


TEMPORAL_PROMPT = """The following media contains consecutive frames from the same live camera stream.
It is ordered from earliest to latest. Use temporal information across the sequence.
Answer ONLY the user's assistive task. Focus on changes near the end of the sequence.
Keep the answer concise for text-to-speech.

User's assistive task: {tool_prompt}"""


def build_multiframe_prompt(tool_prompt: str) -> str:
    concise_prompt = " ".join(str(tool_prompt or "").split())
    return TEMPORAL_PROMPT.format(
        tool_prompt=concise_prompt or "Describe the current scene concisely."
    )


def build_multi_image_request(
    model: str,
    tool_prompt: str,
    frames: Sequence[TimedFrame],
    max_tokens: int,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("At least one frame is required")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_multiframe_prompt(tool_prompt)}
    ]
    content.extend({
        "type": "image_url",
        "image_url": {"url": frame.image_data_uri},
    } for frame in frames)
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
    }


def build_video_request(
    model: str,
    tool_prompt: str,
    video_data_uri: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_multiframe_prompt(tool_prompt)},
                {"type": "video_url", "video_url": {"url": video_data_uri}},
            ],
        }],
        "max_tokens": max_tokens,
        "stream": False,
    }


def _decode_image_data_uri(data_uri: str) -> Image.Image:
    encoded = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
    except Exception as exc:
        raise NvidiaHostedError(f"Could not decode a sampled JPEG frame: {exc}") from exc


def normalize_image_data_uri(data_uri: str, max_edge: int = 768, quality: int = 85) -> str:
    image = _decode_image_data_uri(data_uri)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def encode_frames_as_mp4(
    frames: Sequence[TimedFrame],
    fps: float,
    max_edge: int = 768,
) -> str:
    """CPU-encode sampled frames as an in-memory H.264 MP4 data URI."""
    if not frames:
        raise ValueError("At least one frame is required")
    try:
        import av
    except ImportError as exc:
        raise NvidiaHostedError("PyAV is required for native-video hosted requests") from exc

    images = [_decode_image_data_uri(frame.image_data_uri) for frame in frames]
    first = images[0]
    scale = min(1.0, max_edge / max(first.size))
    width = max(2, int(first.width * scale) // 2 * 2)
    height = max(2, int(first.height * scale) // 2 * 2)
    output = io.BytesIO()
    try:
        with av.open(output, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=max(1, round(fps)))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "25"}
            for image in images:
                resized = image.resize((width, height), Image.Resampling.LANCZOS)
                video_frame = av.VideoFrame.from_ndarray(np.asarray(resized), format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        raise NvidiaHostedError(f"Could not encode sampled frames as MP4: {exc}") from exc
    return "data:video/mp4;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def extract_chat_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise NvidiaHostedError(
            "NVIDIA response did not contain choices[0].message.content"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise NvidiaHostedError("NVIDIA response content was empty or not text")
    return content.strip()


class NvidiaHostedClient:
    def __init__(self, base_url: str, api_key: str, request_timeout_seconds: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.request_timeout_seconds = request_timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    async def _http(self) -> aiohttp.ClientSession:
        if not self.api_key:
            raise NvidiaHostedError("NVIDIA_HOSTED_API_KEY is required")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def list_models(self) -> list[HostedModel]:
        session = await self._http()
        try:
            async with session.get(f"{self.base_url}/models") as response:
                payload = await self._json_response(response, "model discovery")
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise NvidiaHostedError(f"NVIDIA hosted model discovery failed: {exc}") from exc
        models = parse_hosted_models(payload)
        if not models:
            raise NvidiaHostedError("NVIDIA hosted /models returned no model IDs")
        return models

    async def select_model(self, configured_model: str = "") -> HostedModel:
        models = await self.list_models()
        logger.info("[NVIDIA Hosted] available models=%s", [model.id for model in models])
        selected = select_hosted_model(models, configured_model)
        logger.info(
            "[NVIDIA Hosted] selected model=%s vision=%s video=%s",
            selected.id,
            selected.supports_vision,
            selected.supports_video,
        )
        return selected

    async def chat_completions(self, request: dict[str, Any]) -> str:
        session = await self._http()
        try:
            async with session.post(f"{self.base_url}/chat/completions", json=request) as response:
                payload = await self._json_response(response, "chat completion")
        except NvidiaHostedError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise NvidiaHostedError(f"NVIDIA hosted chat completion failed: {exc}") from exc
        return extract_chat_content(payload)

    @staticmethod
    async def _json_response(response: aiohttp.ClientResponse, operation: str) -> Any:
        body = await response.text()
        if response.status >= 400:
            raise NvidiaHostedError(
                f"NVIDIA hosted {operation} failed with HTTP {response.status}: {body}"
            )
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise NvidiaHostedError(
                f"NVIDIA hosted {operation} returned invalid JSON: {body[:1000]}"
            ) from exc
