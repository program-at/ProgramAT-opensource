"""Async client for NVIDIA Real-Time VLM (RTVI) stream APIs."""

from __future__ import annotations

import json
import logging
import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import aiohttp


logger = logging.getLogger(__name__)


class RtviError(RuntimeError):
    """Raised when RTVI rejects or cannot service a request."""


@dataclass(frozen=True)
class RtviCaption:
    content: str
    chunk_id: Any = None
    start_time: Any = None
    end_time: Any = None


def parse_model_ids(payload: Any) -> list[str]:
    """Accept common OpenAI-style and NVIDIA model-list response shapes."""
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("data") or payload.get("models") or payload.get("results") or []
    else:
        entries = []

    model_ids: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            model_id = entry.strip()
        elif isinstance(entry, dict):
            model_id = str(entry.get("id") or entry.get("model") or "").strip()
        else:
            model_id = ""
        if model_id:
            model_ids.append(model_id)
    return model_ids


def parse_stream_registration(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise RtviError("RTVI stream registration returned a non-object response")
    errors = payload.get("errors") or []
    if errors:
        raise RtviError(f"RTVI stream registration failed: {errors}")
    results = payload.get("results") or []
    if not results or not isinstance(results[0], dict) or not results[0].get("id"):
        raise RtviError("RTVI stream registration did not return results[0].id")
    return str(results[0]["id"])


def parse_caption_event(data: str) -> list[RtviCaption]:
    """Parse one SSE data field into zero or more non-empty captions."""
    stripped = data.strip()
    if not stripped or stripped == "[DONE]":
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RtviError(f"Malformed RTVI SSE JSON: {exc}") from exc
    responses = payload.get("chunk_responses") if isinstance(payload, dict) else None
    if not isinstance(responses, list):
        return []
    captions: list[RtviCaption] = []
    for response in responses:
        if not isinstance(response, dict):
            continue
        content = str(response.get("content") or "").strip()
        if content:
            captions.append(RtviCaption(
                content=content,
                chunk_id=response.get("chunk_id"),
                start_time=response.get("start_time"),
                end_time=response.get("end_time"),
            ))
    return captions


class NvidiaRtviClient:
    """Shared-session HTTP client; one instance may serve many ProgramAT clients."""

    def __init__(self, base_url: str, request_timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self._session: aiohttp.ClientSession | None = None
        self._model_id: str | None = None
        self._model_lock = asyncio.Lock()

    def _url(self, path: str) -> str:
        """Join native RTVI paths when base_url is either host or host/v1."""
        if self.base_url.endswith("/v1") and path.startswith("/v1/"):
            return f"{self.base_url}{path[3:]}"
        return f"{self.base_url}{path}"

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def discover_model(self) -> str:
        if self._model_id:
            return self._model_id
        async with self._model_lock:
            if self._model_id:
                return self._model_id
            session = await self._http()
            try:
                async with session.get(self._url("/v1/models")) as response:
                    payload = await self._json_response(response, "model discovery")
            except (aiohttp.ClientError, TimeoutError) as exc:
                raise RtviError(f"RTVI model discovery failed: {exc}") from exc
            models = parse_model_ids(payload)
            if not models:
                raise RtviError("RTVI /v1/models returned no usable model IDs")
            self._model_id = models[0]
            logger.info("[NVIDIA RTVI] resolved model=%s", self._model_id)
            return self._model_id

    async def register_stream(self, rtsp_url: str, description: str) -> str:
        session = await self._http()
        body = {"streams": [{"liveStreamUrl": rtsp_url, "description": description}]}
        try:
            async with session.post(self._url("/v1/streams/add"), json=body) as response:
                payload = await self._json_response(response, "stream registration")
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RtviError(f"RTVI stream registration failed: {exc}") from exc
        return parse_stream_registration(payload)

    async def stream_captions(
        self,
        stream_id: str,
        prompt: str,
        model: str,
        chunk_duration: float,
        chunk_overlap_duration: float,
    ) -> AsyncIterator[RtviCaption]:
        session = await self._http()
        body = {
            "id": stream_id,
            "prompt": prompt,
            "model": model,
            "stream": True,
            "chunk_duration": chunk_duration,
            "chunk_overlap_duration": chunk_overlap_duration,
            "enable_audio": False,
        }
        stream_timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self.request_timeout_seconds,
            sock_connect=self.request_timeout_seconds,
            sock_read=None,
        )
        try:
            async with session.post(
                self._url("/v1/generate_captions"),
                json=body,
                headers={"Accept": "text/event-stream"},
                timeout=stream_timeout,
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:1000]
                    raise RtviError(
                        f"RTVI caption request failed with HTTP {response.status}: {detail}"
                    )
                data_lines: list[str] = []
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        if data_lines:
                            event_data = "\n".join(data_lines)
                            data_lines.clear()
                            if event_data.strip() == "[DONE]":
                                return
                            for caption in parse_caption_event(event_data):
                                yield caption
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines:
                    event_data = "\n".join(data_lines)
                    if event_data.strip() != "[DONE]":
                        for caption in parse_caption_event(event_data):
                            yield caption
        except RtviError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RtviError(f"RTVI caption SSE disconnected: {exc}") from exc

    async def cleanup_stream(self, stream_id: str) -> None:
        session = await self._http()
        endpoints = (
            f"/v1/generate_captions/{stream_id}",
            f"/v1/streams/delete/{stream_id}",
        )
        for endpoint in endpoints:
            try:
                async with session.delete(self._url(endpoint)) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        logger.warning(
                            "[NVIDIA RTVI] cleanup failed endpoint=%s status=%s body=%s",
                            endpoint,
                            response.status,
                            detail,
                        )
            except (aiohttp.ClientError, TimeoutError) as exc:
                logger.warning("[NVIDIA RTVI] cleanup failed endpoint=%s error=%s", endpoint, exc)

    @staticmethod
    async def _json_response(response: aiohttp.ClientResponse, operation: str) -> Any:
        if response.status >= 400:
            detail = (await response.text())[:1000]
            raise RtviError(
                f"RTVI {operation} failed with HTTP {response.status}: {detail}"
            )
        try:
            return await response.json(content_type=None)
        except (json.JSONDecodeError, aiohttp.ContentTypeError) as exc:
            detail = (await response.text())[:1000]
            raise RtviError(f"RTVI {operation} returned invalid JSON: {detail}") from exc
