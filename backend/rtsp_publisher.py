"""Bounded JPEG-to-RTSP publisher used only by the NVIDIA RTVI experiment."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
from dataclasses import dataclass


logger = logging.getLogger(__name__)


def safe_rtsp_path(session_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", session_id).strip("-")
    return (safe or "session")[:96]


def decode_jpeg_data_uri(frame: str) -> bytes:
    encoded = frame.split(",", 1)[1] if frame.startswith("data:") else frame
    return base64.b64decode(encoded, validate=True)


@dataclass(frozen=True)
class RtspPublisherConfig:
    public_base_url: str
    fps: float = 5.0
    width: int | None = None
    height: int | None = None
    ffmpeg_binary: str = "ffmpeg"


class RtspPublisher:
    """One long-lived FFmpeg process with latest-frame queue semantics."""

    def __init__(self, session_id: str, config: RtspPublisherConfig):
        if config.fps <= 0:
            raise ValueError("PROGRAMAT_RTSP_FPS must be greater than zero")
        if bool(config.width) != bool(config.height):
            raise ValueError(
                "PROGRAMAT_RTSP_WIDTH and PROGRAMAT_RTSP_HEIGHT must be set together"
            )
        self.session_id = safe_rtsp_path(session_id)
        self.config = config
        self.rtsp_url = f"{config.public_base_url.rstrip('/')}/{self.session_id}"
        self.process: asyncio.subprocess.Process | None = None
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)
        self._writer_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._first_frame_written = asyncio.Event()
        self.frames_received = 0
        self.frames_dropped = 0

    @property
    def writer_task(self) -> asyncio.Task | None:
        return self._writer_task

    def command(self) -> list[str]:
        command = [
            self.config.ffmpeg_binary,
            "-hide_banner",
            "-loglevel", "warning",
            "-fflags", "nobuffer",
            "-f", "image2pipe",
            "-framerate", str(self.config.fps),
            "-vcodec", "mjpeg",
            "-i", "pipe:0",
        ]
        if self.config.width and self.config.height:
            command.extend(["-vf", f"scale={self.config.width}:{self.config.height}"])
        command.extend([
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-g", str(max(1, round(self.config.fps * 2))),
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ])
        return command

    async def start(self) -> None:
        if self.process is not None:
            return
        resolved_ffmpeg = shutil.which(self.config.ffmpeg_binary)
        if resolved_ffmpeg is None:
            raise RuntimeError(
                f"FFmpeg executable {self.config.ffmpeg_binary!r} was not found. "
                "Install FFmpeg or set PROGRAMAT_FFMPEG_BINARY to its absolute path."
            )
        logger.info(
            "[NVIDIA RTVI] starting RTSP publisher session=%s url=%s fps=%s",
            self.session_id,
            self.rtsp_url,
            self.config.fps,
        )
        try:
            command = self.command()
            command[0] = resolved_ffmpeg
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"Unable to start FFmpeg RTSP publisher: {exc}") from exc
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())
        logger.info(
            "[NVIDIA RTVI] RTSP publisher process started session=%s pid=%s",
            self.session_id,
            self.process.pid,
        )

    def offer_frame(self, frame: str) -> None:
        jpeg = decode_jpeg_data_uri(frame)
        self.frames_received += 1
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(jpeg)

    async def wait_for_first_frame(self, timeout: float = 10.0) -> None:
        await asyncio.wait_for(self._first_frame_written.wait(), timeout=timeout)
        # Give FFmpeg time to complete the RTSP publish handshake, then verify
        # that the process did not immediately fail (for example, unreachable
        # MediaMTX or a rejected path).
        await asyncio.sleep(min(0.25, timeout / 4))
        if self.process is None or self.process.returncode is not None:
            code = None if self.process is None else self.process.returncode
            raise RuntimeError(f"RTSP publisher was not readable (FFmpeg exit={code})")

    async def stop(self) -> None:
        tasks = [task for task in (self._writer_task, self._stderr_task) if task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        process = self.process
        self.process = None
        if process is not None and process.returncode is None:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        logger.info(
            "[NVIDIA RTVI] RTSP publisher stopped session=%s frames_received=%d frames_dropped=%d",
            self.session_id,
            self.frames_received,
            self.frames_dropped,
        )

    async def _writer_loop(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        try:
            latest_jpeg = await self._queue.get()
            frame_interval = 1.0 / self.config.fps
            while True:
                if latest_jpeg is None:
                    return
                if self.process.returncode is not None:
                    raise RuntimeError(
                        f"FFmpeg exited unexpectedly with code {self.process.returncode}"
                    )
                self.process.stdin.write(latest_jpeg)
                await self.process.stdin.drain()
                if not self._first_frame_written.is_set():
                    self._first_frame_written.set()
                    logger.info(
                        "[NVIDIA RTVI] RTSP publisher ready session=%s url=%s",
                        self.session_id,
                        self.rtsp_url,
                    )
                await asyncio.sleep(frame_interval)
                # Keep publishing at a stable cadence. If a newer source frame
                # arrived during the interval it replaces the held frame;
                # otherwise repeat the latest JPEG so the RTSP stream stays live.
                try:
                    latest_jpeg = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
        except asyncio.CancelledError:
            raise
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
            logger.error(
                "[NVIDIA RTVI] FFmpeg publisher failed session=%s error=%s",
                self.session_id,
                exc,
            )
            raise

    async def _stderr_loop(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    return
                logger.warning(
                    "[NVIDIA RTVI] FFmpeg session=%s: %s",
                    self.session_id,
                    line.decode("utf-8", errors="replace").rstrip(),
                )
        except asyncio.CancelledError:
            raise
