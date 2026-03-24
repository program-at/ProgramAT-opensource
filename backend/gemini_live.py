"""
Gemini Live API Session Manager

Uses the google-genai SDK to connect to Gemini's Live API for
custom-GPT-style streaming tool execution.

When a streaming tool has custom_gpt=yes and a gpt_query, the streaming
mode uses this instead of executing tool code on every frame.

API Reference: https://ai.google.dev/gemini-api/docs/live
"""

import asyncio
import base64
import io
import logging

logger = logging.getLogger(__name__)

# Max dimension (width or height) for images sent to Gemini Live.
# Smaller = faster, but too small hurts recognition accuracy.
GEMINI_IMAGE_MAX_DIM = 1024

# Native audio model — must use response_modalities=["AUDIO"] with speechConfig.
# Text is extracted from model turn text parts alongside audio.
GEMINI_LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"


def _downscale_image_base64(image_base64: str, max_dim: int = GEMINI_IMAGE_MAX_DIM) -> str:
    """Downscale a base64 JPEG image so its largest dimension ≤ max_dim.

    Returns the (possibly smaller) base64 string *without* a data URI prefix.
    Falls back to the original if PIL is not available or an error occurs.
    """
    try:
        from PIL import Image

        raw = image_base64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]

        img = Image.open(io.BytesIO(base64.b64decode(raw)))
        w, h = img.size

        if max(w, h) <= max_dim:
            return raw  # already small enough

        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        result = base64.b64encode(buf.getvalue()).decode("ascii")
        logger.debug(f"Downscaled image {w}x{h} -> {new_w}x{new_h}  "
                      f"({len(raw)}→{len(result)} chars)")
        return result
    except Exception as e:
        logger.warning(f"Image downscale failed, using original: {e}")
        # Strip prefix if present and return original
        if image_base64.startswith("data:"):
            return image_base64.split(",", 1)[1]
        return image_base64


class GeminiLiveSession:
    """
    A single Gemini Live session using the google-genai SDK.

    Lifecycle:
        connect() -> send_image_query() / send_followup() -> close()
    """

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
        self._send_lock = asyncio.Lock()  # Prevent concurrent sends corrupting turn state

    async def connect(self):
        """Connect to Gemini Live API using the google-genai SDK."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        logger.info(f"Connecting to Gemini Live API: {self.model}")

        # Create client with v1alpha API version (required for Live API)
        http_options = types.HttpOptions(api_version='v1alpha')
        self._client = genai.Client(api_key=self.api_key, http_options=http_options)

        # Native audio model requires AUDIO response modality
        # Enable output_audio_transcription to get text alongside audio
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_instruction)]
            ) if self.system_instruction else None,
        )

        # Connect — returns async context manager
        self._session_context = self._client.aio.live.connect(
            model=self.model,
            config=config,
        )
        self.session = await self._session_context.__aenter__()
        self.connected = True

        # Start background receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Gemini Live session established")

    async def _receive_loop(self):
        """Background reader that processes responses from Gemini Live."""
        try:
            while self.connected and self.session:
                turn = self.session.receive()
                async for response in turn:
                    if not self.connected:
                        break

                    if response.server_content:
                        sc = response.server_content

                        # Handle model turn parts (audio bytes, possible text/thinking)
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if hasattr(part, 'text') and part.text:
                                    logger.debug(f"Gemini Live text part (thinking): {part.text[:120]}")
                                    self._current_turn_text += part.text

                        # Handle output transcription (spoken words as text)
                        if sc.output_transcription and sc.output_transcription.text:
                            chunk = sc.output_transcription.text
                            self._current_transcript += chunk

                        # Check for turn completion
                        if sc.turn_complete:
                            # Prefer transcript (what was spoken), fall back to text parts
                            result = self._current_transcript.strip() or self._current_turn_text.strip()
                            self._current_turn_text = ""
                            self._current_transcript = ""
                            self._turn_complete_event.set()
                            logger.info(f"Gemini Live turn complete ({len(result)} chars): {result[:150]}")
                            if self._response_handler and result:
                                logger.info(f"Calling response handler with result...")
                                await self._response_handler(result, is_partial=False)
                                logger.info(f"Response handler returned successfully")
                            elif not self._response_handler:
                                logger.warning("No response handler set!")
                            elif not result:
                                logger.warning("Turn complete but empty result")

        except asyncio.CancelledError:
            logger.info("Gemini Live receive loop cancelled")
        except Exception as e:
            if self.connected:
                logger.error(f"Gemini Live receive loop error: {e}")
            self.connected = False

    def set_response_handler(self, handler):
        """
        Set callback for streaming responses.

        handler signature: async def handler(text: str, is_partial: bool)
        - is_partial=True: incremental text chunk
        - is_partial=False: complete turn text
        """
        self._response_handler = handler

    async def send_image_query(self, image_base64: str, query_text: str, mime_type: str = "image/jpeg") -> str:
        """
        Send image + text query via send_client_content. Returns complete response text.
        """
        if not self.connected or not self.session:
            raise ConnectionError("Gemini Live session not connected")

        async with self._send_lock:
            # Downscale image for faster upload and processing
            image_base64 = _downscale_image_base64(image_base64)

            self._current_turn_text = ""
            self._current_transcript = ""
            self._turn_complete_event.clear()

            try:
                await self.session.send_client_content(
                    turns=[{
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                            {"text": query_text},
                        ],
                    }],
                    turn_complete=True,
                )
                logger.info(f"Sent image+query to Gemini Live: {query_text[:80]} (image {len(image_base64)} chars)")
            except Exception as e:
                logger.error(f"Error sending image+query to Gemini Live: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            try:
                await asyncio.wait_for(self._turn_complete_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Gemini Live response timed out after 30s")
                return self._current_transcript.strip() or self._current_turn_text.strip() or "Response timed out"

            return self._current_transcript.strip() or self._current_turn_text.strip()

    async def send_followup(self, text: str) -> str:
        """
        Send text-only follow-up. Model retains context from previous turns.
        """
        if not self.connected or not self.session:
            raise ConnectionError("Gemini Live session not connected")

        async with self._send_lock:
            self._current_turn_text = ""
            self._current_transcript = ""
            self._turn_complete_event.clear()

            await self.session.send_client_content(
                turns=[{
                    "role": "user",
                    "parts": [{"text": text}],
                }],
                turn_complete=True,
            )
            logger.info(f"Sent follow-up to Gemini Live: {text[:80]}")

            try:
                await asyncio.wait_for(self._turn_complete_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Gemini Live follow-up response timed out")
                return self._current_transcript.strip() or self._current_turn_text.strip() or "Response timed out"

            return self._current_transcript.strip() or self._current_turn_text.strip()

    async def close(self):
        """Close the session."""
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
            except Exception as e:
                logger.warning(f"Error closing Gemini Live session context: {e}")
            self._session_context = None
        self.session = None
        logger.info("Gemini Live session closed")


class GeminiLiveManager:
    """
    Manages per-client Gemini Live sessions.
    Used by stream_server when a streaming tool operates in custom_gpt mode.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sessions: dict[str, GeminiLiveSession] = {}
        self._query_tasks: dict[str, asyncio.Task] = {}
        self._paused: dict[str, bool] = {}

    async def start_session(
        self,
        client_id: str,
        system_instruction: str,
        response_handler,
    ) -> GeminiLiveSession:
        """Start a new Gemini Live session for a client."""
        # Stop existing session if any
        await self.stop_session(client_id)

        # Prepend a conciseness directive so the model speaks quickly
        brevity_prefix = (
            "Be extremely concise. Respond in 1-2 short sentences max. "
            "Your responses will be spoken aloud, so keep them brief and direct. "
            "You will receive a series of images from a live camera feed. "
            "Each image is a new independent frame — only describe what you see in the CURRENT image. "
            "Never compare to or reference previous images. Treat each as standalone. "
        )
        full_instruction = brevity_prefix + (system_instruction or "")

        session = GeminiLiveSession(self.api_key, full_instruction)
        session.set_response_handler(response_handler)
        await session.connect()
        self.sessions[client_id] = session

        logger.info(f"Started Gemini Live session for {client_id}")
        return session

    async def run_query_loop(
        self,
        client_id: str,
        query_text: str,
        get_current_frame,
        interval_seconds: float = 5.0,
    ):
        """
        Periodically re-send the gpt_query with a fresh camera frame.
        This is the custom-GPT behavior within streaming mode.
        Skips queries while the session is paused (e.g. during follow-up).

        Args:
            client_id: Client identifier
            query_text: The query to re-ask (gpt_query from tool spec)
            get_current_frame: Callable returning (image_base64, image_cv2) or (None, None)
            interval_seconds: How often to re-query (default 5s)
        """
        session = self.sessions.get(client_id)
        if not session:
            logger.error(f"No Gemini Live session for {client_id}")
            return

        logger.info(
            f"Starting live query loop for {client_id}: "
            f"'{query_text[:60]}' every {interval_seconds}s"
        )

        try:
            while session.connected and client_id in self.sessions:
                # Skip query if paused (user is asking a follow-up)
                if self._paused.get(client_id, False):
                    logger.debug(f"Query loop paused for {client_id}, skipping")
                    await asyncio.sleep(0.5)
                    continue

                try:
                    frame_base64, _ = get_current_frame()
                    if frame_base64:
                        logger.info(f"Query loop sending image ({len(frame_base64)} chars) for {client_id}")
                        response_text = await session.send_image_query(frame_base64, query_text)
                        logger.info(f"Query loop got response ({len(response_text)} chars) for {client_id}: {response_text[:100]}")
                    else:
                        logger.warning(f"No frame available for {client_id}, skipping query")
                except ConnectionError:
                    logger.warning(f"Gemini Live connection lost for {client_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in live query loop for {client_id}: {e}")

                # Fixed sleep AFTER the response so the next query uses a fresh frame
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info(f"Live query loop cancelled for {client_id}")

    def pause_query_loop(self, client_id: str):
        """Pause the periodic query loop for a client (e.g. during follow-up)."""
        self._paused[client_id] = True
        logger.info(f"Paused query loop for {client_id}")

    def resume_query_loop(self, client_id: str):
        """Resume the periodic query loop for a client."""
        self._paused[client_id] = False
        logger.info(f"Resumed query loop for {client_id}")

    async def send_followup(self, client_id: str, text: str) -> str:
        """Send a follow-up question to an active session."""
        session = self.sessions.get(client_id)
        if not session or not session.connected:
            raise ConnectionError(f"No active Gemini Live session for {client_id}")
        return await session.send_followup(text)

    async def stop_session(self, client_id: str):
        """Stop and clean up a client's session."""
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
            logger.info(f"Stopped Gemini Live session for {client_id}")

    async def stop_all(self):
        """Stop all active sessions."""
        for client_id in list(self.sessions.keys()):
            await self.stop_session(client_id)
