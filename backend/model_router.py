"""Central capability-based routing for all LLM interactions."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False


logger = logging.getLogger(__name__)

MODEL_PROFILES_PATH = Path(__file__).with_name("model_profiles.yaml")
CAPABILITY_PROFILES_PATH = Path(__file__).with_name("capability_profiles.yaml")

DEFAULT_MODEL_PROFILES: Dict[str, Any] = {
    "models": {
        "gemini_flash": {
            "model": "gemini/gemini-3-flash-preview",
            "vision": 4,
            "coding": 3,
            "reasoning": 2,
            "latency": 5,
            "cost": 5,
        },
        "claude": {
            "model": "anthropic/claude-3-5-sonnet-20241022",
            "vision": 2,
            "coding": 5,
            "reasoning": 5,
            "latency": 2,
            "cost": 2,
        },
        "gpt4o": {
            "model": "openai/gpt-4o",
            "vision": 5,
            "coding": 4,
            "reasoning": 4,
            "latency": 3,
            "cost": 1,
        },
    }
}

DEFAULT_CAPABILITY_PROFILES: Dict[str, Any] = {
    "capabilities": {
        "text_parse": {"vision": 0, "coding": 0, "reasoning": 1, "latency": 5},
        "tool_retrieval": {"vision": 0, "coding": 0, "reasoning": 2, "latency": 5},
        "image_analysis": {"vision": 5, "coding": 0, "reasoning": 2, "latency": 4},
        "code_generation": {"vision": 0, "coding": 5, "reasoning": 4, "latency": 2},
        "code_repair": {"vision": 0, "coding": 5, "reasoning": 5, "latency": 2},
    }
}

CAPABILITY_TO_PROFILE = {
    "text_parse": "gemini_flash",
    "tool_retrieval": "gemini_flash",
    "image_analysis": "gpt4o",
    "code_generation": "claude",
    "code_repair": "claude",
    "summarization": "gemini_flash",
}

_MODEL_PROFILES: Dict[str, Any] | None = None
_CAPABILITY_PROFILES: Dict[str, Any] | None = None
ROUTING_MODE = os.environ.get("ROUTING_MODE", "fixed").strip().lower()


def _routing_mode() -> str:
    return ROUTING_MODE if ROUTING_MODE in {"fixed", "score"} else "fixed"


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return fallback

    try:
        parsed: Dict[str, Any] = {}
        section_data: Optional[Dict[str, Any]] = None
        item_data: Optional[Dict[str, Any]] = None

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            if indent == 0 and stripped.endswith(":"):
                section_data = parsed.setdefault(stripped[:-1].strip(), {})
                item_data = None
                continue

            if indent == 2 and stripped.endswith(":") and section_data is not None:
                item_data = section_data.setdefault(stripped[:-1].strip(), {})
                continue

            if indent >= 4 and ":" in stripped and item_data is not None:
                key, raw_value = stripped.split(":", 1)
                item_data[key.strip()] = _parse_scalar(raw_value)

        return parsed or fallback
    except Exception as exc:
        logger.warning(f"Failed to parse {path}: {exc}")
        return fallback


def _model_profiles() -> Dict[str, Any]:
    global _MODEL_PROFILES
    if _MODEL_PROFILES is None:
        _MODEL_PROFILES = _load_simple_yaml(MODEL_PROFILES_PATH, DEFAULT_MODEL_PROFILES)
    return _MODEL_PROFILES


def _capability_profiles() -> Dict[str, Any]:
    global _CAPABILITY_PROFILES
    if _CAPABILITY_PROFILES is None:
        _CAPABILITY_PROFILES = _load_simple_yaml(CAPABILITY_PROFILES_PATH, DEFAULT_CAPABILITY_PROFILES)
    return _CAPABILITY_PROFILES


def _provider_for_model(model_name: str) -> str:
    raw = (model_name or "").strip()
    if "/" in raw:
        return raw.split("/", 1)[0]
    if raw.startswith("gemini"):
        return "gemini"
    if raw.startswith("claude"):
        return "anthropic"
    if raw.startswith("gpt"):
        return "openai"
    return "unknown"


def _resolve_explicit_profile(metadata: Dict[str, Any] | None) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None

    candidate = metadata.get("requested_profile") or metadata.get("profile")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    requested_model = metadata.get("requested_model") or metadata.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        return None

    requested_model = requested_model.strip()
    for profile_name, profile_data in _model_profiles().get("models", {}).items():
        model_name = str(profile_data.get("model", "")).strip()
        short_model_name = model_name.split("/", 1)[1] if "/" in model_name else model_name
        if (
            requested_model == profile_name
            or requested_model == model_name
            or requested_model == short_model_name
            or requested_model == profile_name.replace("_", "-")
        ):
            return profile_name
    return None


def get_route_info(capability: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    models = _model_profiles().get("models", {})
    capabilities = _capability_profiles().get("capabilities", {})

    capability_key = (capability or "").strip().lower()
    routing_mode = _routing_mode()
    capability_profile = capabilities.get(capability_key, {})

    if routing_mode == "score" and capability_profile:
        selected_profile, scoreboard = _select_scored_profile(capability_key)
    else:
        selected_profile = _resolve_explicit_profile(metadata) or CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash")
        scoreboard = None

    profile_data = models.get(selected_profile, {})
    selected_model = str(profile_data.get("model", "")).strip()

    if not selected_model:
        selected_profile = "gemini_flash"
        profile_data = models.get(selected_profile, {})
        selected_model = str(profile_data.get("model", "gemini/gemini-2.0-flash")).strip()
        routing_mode = "fixed"

    if routing_mode == "score" and scoreboard is not None:
        _log_scoreboard(capability_key, scoreboard, selected_profile)

    return {
        "capability": capability_key,
        "selected_profile": selected_profile,
        "selected_model": selected_model,
        "provider": _provider_for_model(selected_model),
        "profile_data": profile_data,
        "capability_profile": capability_profile,
        "routing_mode": routing_mode,
    }


def score(model: Dict[str, Any], capability: Dict[str, Any]) -> float:
    vision_weight = float(capability.get("vision", 0) or 0)
    coding_weight = float(capability.get("coding", 0) or 0)
    reasoning_weight = float(capability.get("reasoning", 0) or 0)
    latency_weight = float(capability.get("latency", 0) or 0)

    model_vision = float(model.get("vision", 0) or 0)
    model_coding = float(model.get("coding", 0) or 0)
    model_reasoning = float(model.get("reasoning", 0) or 0)
    model_latency = float(model.get("latency", 0) or 0)

    return (
        vision_weight * min(model_vision, float(capability.get("vision", 0) or 0))
        + coding_weight * min(model_coding, float(capability.get("coding", 0) or 0))
        + reasoning_weight * min(model_reasoning, float(capability.get("reasoning", 0) or 0))
        + latency_weight * model_latency
    )


def _select_scored_profile(capability_key: str) -> tuple[str, Dict[str, float]]:
    models = _model_profiles().get("models", {})
    capability_profile = _capability_profiles().get("capabilities", {}).get(capability_key, {})

    if not models:
        return CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash"), {}

    scores: Dict[str, float] = {}
    best_profile = None
    best_score = float("-inf")

    for profile_name, model_profile in models.items():
        model_score = score(model_profile, capability_profile)
        scores[profile_name] = model_score
        if model_score > best_score:
            best_profile = profile_name
            best_score = model_score

    if best_profile is None:
        best_profile = CAPABILITY_TO_PROFILE.get(capability_key, "gemini_flash")

    return best_profile, scores


def _log_scoreboard(capability_key: str, scoreboard: Dict[str, float], selected_profile: str) -> None:
    if not scoreboard:
        return

    lines = ["[ROUTER SCORE]", f"capability={capability_key}"]
    for profile_name in _model_profiles().get("models", {}).keys():
        if profile_name in scoreboard:
            lines.append(f"{profile_name}={scoreboard[profile_name]:.1f}")
    lines.append(f"selected={selected_profile}")
    logger.info("\n".join(lines))


def get_selected_model(capability: str, metadata: Dict[str, Any] | None = None) -> str:
    return get_route_info(capability, metadata)["selected_model"]


def _resolve_api_key(model_name: str, metadata: Dict[str, Any] | None = None) -> str:
    if isinstance(metadata, dict):
        explicit_api_key = metadata.get("api_key") or metadata.get("explicit_api_key")
        if isinstance(explicit_api_key, str) and explicit_api_key.strip():
            return explicit_api_key.strip()

    normalized = (model_name or "").lower()
    if normalized.startswith("gemini"):
        return os.environ.get("GEMINI_API_KEY", "")
    if normalized.startswith("groq"):
        return os.environ.get("GROQ_API_KEY", "")
    if normalized.startswith("anthropic") or normalized.startswith("claude"):
        return os.environ.get("ANTHROPIC_API_KEY", "")
    if normalized.startswith("openai") or normalized.startswith("gpt"):
        return os.environ.get("OPENAI_API_KEY", "")
    return os.environ.get("GROQ_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


def _image_to_data_uri(image: Any) -> str:
    if isinstance(image, str):
        raw = image.strip()
        return raw if raw.startswith("data:") else f"data:image/jpeg;base64,{raw}"

    if isinstance(image, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(image)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        pass

    raise TypeError(f"Unsupported image type for llm_call: {type(image).__name__}")


def _merge_messages(messages: List[Dict[str, Any]], images: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    merged = [dict(message) for message in (messages or [])]
    image_items = list(images or [])
    if not image_items:
        return merged

    image_parts = [{"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}} for image in image_items]

    target_index = None
    for index in range(len(merged) - 1, -1, -1):
        if merged[index].get("role") == "user":
            target_index = index
            break

    if target_index is None:
        merged.append({"role": "user", "content": image_parts})
        return merged

    content = merged[target_index].get("content")
    if isinstance(content, list):
        merged[target_index]["content"] = list(content) + image_parts
    elif isinstance(content, str) and content.strip():
        merged[target_index]["content"] = [{"type": "text", "text": content}] + image_parts
    else:
        merged[target_index]["content"] = image_parts

    return merged


def llm_call(
    capability: str,
    messages: List[Dict[str, Any]],
    images: Optional[Iterable[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Route a request through the fixed capability map and call LiteLLM."""
    if not LITELLM_AVAILABLE:
        raise ImportError("litellm is not available")

    route_info = get_route_info(capability, metadata)
    selected_profile = route_info["selected_profile"]
    selected_model = route_info["selected_model"]
    provider = route_info["provider"]

    logger.info(
        f"[ROUTER] capability={capability} selected_profile={selected_profile} selected_model={selected_model} provider={provider}"
    )

    completion_kwargs: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        for key in ("temperature", "max_tokens", "top_p", "stop", "timeout", "response_format", "stream"):
            if key in metadata and metadata[key] is not None:
                completion_kwargs[key] = metadata[key]

    payload = _merge_messages(messages, images)
    api_key = _resolve_api_key(selected_model, metadata)

    try:
        return litellm.completion(
            model=selected_model,
            messages=payload,
            api_key=api_key,
            **completion_kwargs,
        )
    except Exception:
        logger.exception(
            f"[ROUTER] capability={capability} selected_profile={selected_profile} selected_model={selected_model} provider={provider}"
        )
        raise


def _downscale_image_base64(image_base64: str, max_dim: int = 1024) -> str:
    try:
        from PIL import Image

        raw = image_base64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]

        img = Image.open(io.BytesIO(base64.b64decode(raw)))
        width, height = img.size
        if max(width, height) <= max_dim:
            return raw

        scale = max_dim / max(width, height)
        new_width, new_height = int(width * scale), int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        result = base64.b64encode(buffer.getvalue()).decode("ascii")
        logger.debug(f"Downscaled image {width}x{height} -> {new_width}x{new_height}")
        return result
    except Exception as exc:
        logger.warning(f"Image downscale failed, using original: {exc}")
        return image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64


GEMINI_IMAGE_MAX_DIM = 1024
GEMINI_LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"


class GeminiLiveSession:
    def __init__(self, api_key: str, system_instruction: str = "", model: str = None):
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.model = model or GEMINI_LIVE_MODEL
        self.session = None
        self._session_context = None
        self._client = None
        self.connected = False
        self._response_handler = None
        self._receive_task = None
        self._current_turn_text = ""
        self._current_transcript = ""
        self._turn_complete_event = asyncio.Event()
        self._send_lock = asyncio.Lock()

    async def connect(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        logger.info(f"Connecting to Gemini Live API: {self.model}")
        http_options = types.HttpOptions(api_version="v1alpha")
        self._client = genai.Client(api_key=self.api_key, http_options=http_options)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(parts=[types.Part(text=self.system_instruction)]) if self.system_instruction else None,
        )
        self._session_context = self._client.aio.live.connect(model=self.model, config=config)
        self.session = await self._session_context.__aenter__()
        self.connected = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Gemini Live session established")

    async def _receive_loop(self):
        try:
            while self.connected and self.session:
                turn = self.session.receive()
                async for response in turn:
                    if not self.connected:
                        break
                    if response.server_content:
                        sc = response.server_content
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if hasattr(part, "text") and part.text:
                                    self._current_turn_text += part.text
                        if sc.output_transcription and sc.output_transcription.text:
                            self._current_transcript += sc.output_transcription.text
                        if sc.turn_complete:
                            result = self._current_transcript.strip() or self._current_turn_text.strip()
                            self._current_turn_text = ""
                            self._current_transcript = ""
                            self._turn_complete_event.set()
                            if self._response_handler and result:
                                await self._response_handler(result, is_partial=False)
        except asyncio.CancelledError:
            logger.info("Gemini Live receive loop cancelled")
        except Exception as exc:
            if self.connected:
                logger.error(f"Gemini Live receive loop error: {exc}")
            self.connected = False

    def set_response_handler(self, handler):
        self._response_handler = handler

    async def send_image_query(self, image_base64: str, query_text: str, mime_type: str = "image/jpeg") -> str:
        if not self.connected or not self.session:
            raise ConnectionError("Gemini Live session not connected")

        async with self._send_lock:
            image_base64 = _downscale_image_base64(image_base64, GEMINI_IMAGE_MAX_DIM)
            self._current_turn_text = ""
            self._current_transcript = ""
            self._turn_complete_event.clear()
            await self.session.send_client_content(
                turns=[{"role": "user", "parts": [{"inline_data": {"mime_type": mime_type, "data": image_base64}}, {"text": query_text}]}],
                turn_complete=True,
            )
            try:
                await asyncio.wait_for(self._turn_complete_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                return self._current_transcript.strip() or self._current_turn_text.strip() or "Response timed out"
            return self._current_transcript.strip() or self._current_turn_text.strip()

    async def send_followup(self, text: str) -> str:
        if not self.connected or not self.session:
            raise ConnectionError("Gemini Live session not connected")

        async with self._send_lock:
            self._current_turn_text = ""
            self._current_transcript = ""
            self._turn_complete_event.clear()
            await self.session.send_client_content(turns=[{"role": "user", "parts": [{"text": text}]}], turn_complete=True)
            try:
                await asyncio.wait_for(self._turn_complete_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                return self._current_transcript.strip() or self._current_turn_text.strip() or "Response timed out"
            return self._current_transcript.strip() or self._current_turn_text.strip()

    async def close(self):
        self.connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning(f"Error closing Gemini Live session context: {exc}")
            self._session_context = None
        self.session = None
        logger.info("Gemini Live session closed")


class GeminiLiveManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sessions: dict[str, GeminiLiveSession] = {}
        self._query_tasks: dict[str, asyncio.Task] = {}
        self._paused: dict[str, bool] = {}

    async def start_session(self, client_id: str, system_instruction: str, response_handler) -> GeminiLiveSession:
        await self.stop_session(client_id)
        brevity_prefix = (
            "Be extremely concise. Respond in 1-2 short sentences max. "
            "Your responses will be spoken aloud, so keep them brief and direct. "
            "You will receive a series of images from a live camera feed. "
            "Each image is a new independent frame — only describe what you see in the CURRENT image. "
            "Never compare to or reference previous images. Treat each as standalone. "
        )
        session = GeminiLiveSession(self.api_key, brevity_prefix + (system_instruction or ""))
        session.set_response_handler(response_handler)
        await session.connect()
        self.sessions[client_id] = session
        return session

    async def run_query_loop(self, client_id: str, query_text: str, get_current_frame, interval_seconds: float = 5.0):
        session = self.sessions.get(client_id)
        if not session:
            logger.error(f"No Gemini Live session for {client_id}")
            return

        try:
            while session.connected and client_id in self.sessions:
                if self._paused.get(client_id, False):
                    await asyncio.sleep(0.5)
                    continue

                try:
                    frame_base64, _ = get_current_frame()
                    if frame_base64:
                        await session.send_image_query(frame_base64, query_text)
                except ConnectionError:
                    break
                except Exception as exc:
                    logger.error(f"Error in live query loop for {client_id}: {exc}")

                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info(f"Live query loop cancelled for {client_id}")

    def pause_query_loop(self, client_id: str):
        self._paused[client_id] = True

    def resume_query_loop(self, client_id: str):
        self._paused[client_id] = False

    async def send_followup(self, client_id: str, text: str) -> str:
        session = self.sessions.get(client_id)
        if not session or not session.connected:
            raise ConnectionError(f"No active Gemini Live session for {client_id}")
        return await session.send_followup(text)

    async def stop_session(self, client_id: str):
        self._paused.pop(client_id, None)
        if client_id in self._query_tasks:
            self._query_tasks[client_id].cancel()
            try:
                await self._query_tasks[client_id]
            except asyncio.CancelledError:
                pass
            del self._query_tasks[client_id]
        if client_id in self.sessions:
            await self.sessions[client_id].close()
            del self.sessions[client_id]

    async def stop_all(self):
        for client_id in list(self.sessions.keys()):
            await self.stop_session(client_id)


__all__ = [
    "llm_call",
    "get_route_info",
    "get_selected_model",
    "GeminiLiveSession",
    "GeminiLiveManager",
    "GEMINI_IMAGE_MAX_DIM",
    "GEMINI_LIVE_MODEL",
]