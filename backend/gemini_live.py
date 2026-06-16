"""Gemini Live session management for ProgramAT live camera interactions."""

from __future__ import annotations

import asyncio
import base64
import io
import logging


logger = logging.getLogger(__name__)

GEMINI_IMAGE_MAX_DIM = 1024
GEMINI_LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"


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
        logger.debug("Downscaled image %sx%s -> %sx%s", width, height, new_width, new_height)
        return result
    except Exception as exc:
        logger.warning("Image downscale failed, using original: %s", exc)
        return image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64


class GeminiLiveSession:
    def __init__(self, api_key: str, system_instruction: str = "", model: str | None = None):
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

        logger.info("Connecting to Gemini Live API: %s", self.model)
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
                logger.error("Gemini Live receive loop error: %s", exc)
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
                logger.warning("Error closing Gemini Live session context: %s", exc)
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
            "Each image is a new independent frame -- only describe what you see in the CURRENT image. "
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
            logger.error("No Gemini Live session for %s", client_id)
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
                    logger.error("Error in live query loop for %s: %s", client_id, exc)

                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Live query loop cancelled for %s", client_id)

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
    "GeminiLiveSession",
    "GeminiLiveManager",
    "GEMINI_IMAGE_MAX_DIM",
    "GEMINI_LIVE_MODEL",
]
