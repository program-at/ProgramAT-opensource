"""NVIDIA-hosted OpenAI-compatible VLM client and multiframe helpers."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import aiohttp
import numpy as np
from PIL import Image, ImageOps


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
                {
                    "type": "text",
                    "text": (
                        "This is a chronological clip representing the most recent "
                        "configured video window. Answer according to the tool prompt below.\n\n"
                        + tool_prompt
                    ),
                },
                {"type": "video_url", "video_url": {"url": video_data_uri}},
            ],
        }],
        "max_tokens": max_tokens,
        "stream": False,
    }


def _decode_image_data_uri(data_uri: str) -> Image.Image:
    encoded = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
        return ImageOps.exif_transpose(image).convert("RGB")
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


def encode_frames_to_mp4(
    frames: Sequence[TimedFrame],
    output_path: str | Path,
    fps: float = 4,
    max_width: int = 1280,
    jpeg_quality: int = 80,
    ffmpeg_binary: str = "ffmpeg",
) -> int:
    """Encode chronological JPEG frames to a finalized H.264 MP4 with FFmpeg."""
    if not frames:
        raise NvidiaHostedError("At least one frame is required to encode a video clip")
    executable = shutil.which(ffmpeg_binary)
    if executable is None:
        raise NvidiaHostedError(
            f"FFmpeg executable {ffmpeg_binary!r} was not found; install it or set "
            "PROGRAMAT_FFMPEG_BINARY"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="programat-hosted-video-frames-") as frame_dir:
        directory = Path(frame_dir)
        for index, frame in enumerate(sorted(frames, key=lambda item: item.timestamp)):
            try:
                image = _decode_image_data_uri(frame.image_data_uri)
                if image.width > max_width:
                    image.thumbnail((max_width, max_width * 10), Image.Resampling.LANCZOS)
                image.save(
                    directory / f"frame-{index:06d}.jpg", "JPEG",
                    quality=jpeg_quality, optimize=True,
                )
            except Exception as exc:
                raise NvidiaHostedError(f"Could not decode input frame {index}: {exc}") from exc
        scale = f"scale='min({max_width},iw)':-2:force_original_aspect_ratio=decrease"
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(directory / "frame-%06d.jpg"), "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-vf", scale,
            "-movflags", "+faststart", "-y", str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise NvidiaHostedError(f"FFmpeg MP4 encoding failed: {exc.stderr[-1000:]}") from exc
    size = output.stat().st_size
    if size <= 0:
        raise NvidiaHostedError("FFmpeg produced an empty MP4")
    return size


def video_file_data_uri(video_path: str | Path, max_bytes: int) -> str:
    data = Path(video_path).read_bytes()
    if len(data) > max_bytes:
        raise NvidiaHostedError(
            f"MP4 clip is {len(data)} bytes, exceeding base64 limit {max_bytes}"
        )
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def probe_mp4(video_path: str | Path, ffmpeg_binary: str = "ffmpeg") -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        candidate = Path(shutil.which(ffmpeg_binary) or "").with_name("ffprobe")
        ffprobe = str(candidate) if candidate.is_file() else None
    if ffprobe is None:
        raise NvidiaHostedError("ffprobe was not found beside FFmpeg or on PATH")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration,size",
        "-of", "json", str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise NvidiaHostedError(f"Could not inspect encoded MP4: {exc}") from exc


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


def parse_structured_video_result(raw_content: str) -> dict[str, Any]:
    """Parse the exact JSON object returned by a structured streaming tool."""
    text = str(raw_content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NvidiaHostedError(f"Hosted video response was not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise NvidiaHostedError("Hosted video response must be one JSON object")
    return value


def validate_played_card_event(
    parsed: dict[str, Any], confidence_threshold: float = 0.8,
    allow_remaining_cards_in_evidence: bool = False,
) -> tuple[str | None, str]:
    """Deterministically enforce before/after evidence without another model."""
    required_fields = {
        "event_detected", "before_cards", "after_cards", "played_card",
        "confidence", "evidence",
    }
    if set(parsed) != required_fields:
        return None, "invalid_schema_fields"
    before = parsed.get("before_cards")
    after = parsed.get("after_cards")
    played = parsed.get("played_card")
    confidence = parsed.get("confidence")
    evidence = parsed.get("evidence")
    if not isinstance(parsed.get("event_detected"), bool):
        return None, "invalid_event_detected"
    if not isinstance(before, list) or not all(isinstance(item, str) for item in before):
        return None, "invalid_before_cards"
    if not isinstance(after, list) or not all(isinstance(item, str) for item in after):
        return None, "invalid_after_cards"
    if not isinstance(evidence, str) or not evidence.strip():
        return None, "invalid_evidence"
    if isinstance(confidence, bool):
        return None, "invalid_confidence"
    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError):
        return None, "invalid_confidence"
    if not 0.0 <= numeric_confidence <= 1.0:
        return None, "invalid_confidence"
    if numeric_confidence < confidence_threshold:
        return None, "confidence_below_threshold"
    normalize = lambda value: " ".join(value.casefold().split())
    ranks = (
        "ace", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "jack", "queen", "king",
    )
    suits = ("clubs", "diamonds", "hearts", "spades")
    valid_cards = {f"{rank} of {suit}" for rank in ranks for suit in suits}
    before_names = [normalize(item) for item in before]
    after_names = [normalize(item) for item in after]
    if any(name not in valid_cards for name in before_names):
        return None, "invalid_card_in_before"
    if any(name not in valid_cards for name in after_names):
        return None, "invalid_card_in_after"
    if len(before_names) != len(set(before_names)):
        return None, "duplicate_card_in_before"
    if len(after_names) != len(set(after_names)):
        return None, "duplicate_card_in_after"
    before_keys = set(before_names)
    after_keys = set(after_names)
    if parsed.get("event_detected") is not True:
        if played is not None:
            return None, "no_event_played_card_must_be_null"
        if before_keys and before_keys == after_keys:
            return None, "stable_no_event"
        return None, "event_not_detected"
    if not isinstance(played, str) or not played.strip():
        return None, "invalid_played_card"
    played_key = normalize(played)
    if played_key not in valid_cards:
        return None, "invalid_played_card_name"
    if played_key not in before_keys:
        return None, "played_card_not_in_before"
    if played_key in after_keys:
        return None, "played_card_still_in_after"
    evidence_key = normalize(evidence)
    named_in_evidence = {card for card in valid_cards if card in evidence_key}
    if played_key not in named_in_evidence:
        return None, "evidence_card_mismatch"
    if allow_remaining_cards_in_evidence:
        removal_words = r"(?:played|removed|disappeared|absent|taken|left the hand)"
        for card in named_in_evidence - {played_key}:
            card_pattern = re.escape(card)
            if (re.search(
                    rf"{card_pattern}\s+(?:(?:was|is|has been)\s+)?{removal_words}",
                    evidence_key,
                ) or re.search(
                    rf"{removal_words}\s+(?:the\s+)?{card_pattern}", evidence_key,
                )):
                return None, "evidence_identifies_different_removed_card"
    elif named_in_evidence != {played_key}:
        return None, "evidence_card_mismatch"
    return f"You just played the {played_key}.", "accepted"


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
        content, _usage = await self.chat_completions_with_metadata(request)
        return content

    async def chat_completions_with_metadata(
        self, request: dict[str, Any]
    ) -> tuple[str, Any]:
        session = await self._http()
        try:
            async with session.post(f"{self.base_url}/chat/completions", json=request) as response:
                payload = await self._json_response(response, "chat completion")
        except NvidiaHostedError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise NvidiaHostedError(f"NVIDIA hosted chat completion failed: {exc}") from exc
        return extract_chat_content(payload), payload.get("usage") if isinstance(payload, dict) else None

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
