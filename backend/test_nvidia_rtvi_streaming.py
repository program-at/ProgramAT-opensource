"""Offline tests for the experimental NVIDIA RTVI streaming boundary."""

import asyncio
import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stream_server
from nvidia_rtvi_client import (
    RtviCaption,
    RtviError,
    parse_caption_event,
    parse_model_ids,
    parse_stream_registration,
)
from rtsp_publisher import RtspPublisher, RtspPublisherConfig, safe_rtsp_path


class TestRtviResponseParsing(unittest.TestCase):
    def test_model_discovery_accepts_openai_and_nvidia_shapes(self):
        self.assertEqual(parse_model_ids({"data": [{"id": "model-a"}]}), ["model-a"])
        self.assertEqual(
            parse_model_ids({"models": [{"model": "model-b"}, "model-c"]}),
            ["model-b", "model-c"],
        )

    def test_registration_checks_errors_and_stream_id(self):
        self.assertEqual(
            parse_stream_registration({"results": [{"id": "stream-1"}], "errors": []}),
            "stream-1",
        )
        with self.assertRaisesRegex(RtviError, "registration failed"):
            parse_stream_registration({"results": [], "errors": [{"message": "bad URL"}]})
        with self.assertRaisesRegex(RtviError, r"results\[0\]\.id"):
            parse_stream_registration({"results": [], "errors": []})

    def test_sse_caption_parser_keeps_non_empty_chunk_content(self):
        event = json.dumps({
            "chunk_responses": [
                {"chunk_id": 3, "start_time": 4, "end_time": 6, "content": " A door opens. "},
                {"chunk_id": 4, "content": ""},
            ]
        })
        self.assertEqual(
            parse_caption_event(event),
            [RtviCaption("A door opens.", 3, 4, 6)],
        )
        self.assertEqual(parse_caption_event("[DONE]"), [])
        with self.assertRaisesRegex(RtviError, "Malformed RTVI SSE JSON"):
            parse_caption_event("{not json")


class TestRtspPublisher(unittest.TestCase):
    def test_safe_path_and_ffmpeg_low_latency_command(self):
        self.assertEqual(safe_rtsp_path("10.0.0.1:123/a b"), "10-0-0-1-123-a-b")
        publisher = RtspPublisher(
            "client/session",
            RtspPublisherConfig(
                public_base_url="rtsp://media:8554/programat/",
                fps=5,
                width=640,
                height=360,
            ),
        )
        command = publisher.command()
        self.assertEqual(publisher.rtsp_url, "rtsp://media:8554/programat/client-session")
        self.assertIn("libx264", command)
        self.assertIn("zerolatency", command)
        self.assertIn("scale=640:360", command)

    def test_latest_frame_queue_is_bounded_and_counts_drops(self):
        publisher = RtspPublisher(
            "client",
            RtspPublisherConfig(public_base_url="rtsp://media:8554/programat"),
        )
        first = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-one").decode()
        second = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-two").decode()
        publisher.offer_frame(first)
        publisher.offer_frame(second)
        self.assertEqual(publisher._queue.qsize(), 1)
        self.assertEqual(publisher.frames_received, 2)
        self.assertEqual(publisher.frames_dropped, 1)
        self.assertEqual(publisher._queue.get_nowait(), b"jpeg-two")


class TestRtviStreamingBoundary(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        stream_server.active_streaming_tools.clear()
        stream_server.active_streaming_tasks.clear()
        stream_server.active_rtvi_sessions.clear()

    async def test_feature_flag_dispatch_bypasses_programat_scheduler(self):
        stream_server.active_rtvi_sessions["client"] = {"session_id": "generation-a"}
        with patch.object(stream_server, "NVIDIA_STREAMING_MODE", "rtvi"), \
             patch.object(stream_server, "_offer_rtvi_frame", new=AsyncMock()) as offer, \
             patch.object(stream_server, "schedule_streaming_frame", new=AsyncMock()) as schedule:
            task = await stream_server._dispatch_active_streaming_frame(
                "ws", "client", "decoded-image", "jpeg-base64"
            )
        self.assertIsNone(task)
        offer.assert_awaited_once_with("ws", "client", "jpeg-base64")
        schedule.assert_not_awaited()

    async def test_disabled_flag_uses_original_scheduler(self):
        stream_server.active_streaming_tools["client"] = {"tool": {"name": "tool"}}
        with patch.object(stream_server, "NVIDIA_STREAMING_MODE", "original"), \
             patch.object(stream_server, "schedule_streaming_frame", new=AsyncMock()) as schedule:
            task = await stream_server._dispatch_active_streaming_frame(
                "ws", "client", "decoded-image", "jpeg-base64"
            )
            await task
        schedule.assert_awaited_once_with("ws", "client", "decoded-image", "jpeg-base64")

    async def test_duplicate_chunk_ids_are_forwarded_once(self):
        websocket = MagicMock()
        websocket.send = AsyncMock()
        publisher = MagicMock()
        publisher.wait_for_first_frame = AsyncMock()
        session = {
            "session_id": "generation-a",
            "tool_name": "door detector",
            "prompt": "Describe the doorway",
            "rtsp_url": "rtsp://media/programat/generation-a",
            "publisher": publisher,
            "rtvi_stream_id": None,
            "started_at": asyncio.get_running_loop().time(),
            "first_result_at": None,
            "forwarded_chunk_ids": set(),
            "output_sequence": 0,
        }
        stream_server.active_rtvi_sessions["client"] = session

        async def captions(**_kwargs):
            yield RtviCaption("Door ahead", 7, 0, 2)
            yield RtviCaption("Door ahead", 7, 0, 2)

        with patch.object(stream_server, "NVIDIA_RTVI_MODEL", "rtvi-model"), \
             patch.object(stream_server.rtvi_client, "register_stream", new=AsyncMock(return_value="stream-1")), \
             patch.object(stream_server.rtvi_client, "stream_captions", side_effect=captions), \
             patch.object(stream_server, "_cleanup_rtvi_session", new=AsyncMock()) as cleanup:
            await stream_server._run_rtvi_caption_session(websocket, "client", session)

        self.assertEqual(websocket.send.await_count, 1)
        payload = json.loads(websocket.send.await_args.args[0])
        self.assertEqual(payload["result"], "Door ahead")
        self.assertEqual(payload["rtvi_chunk_id"], 7)
        cleanup.assert_awaited_once_with("client", reason="caption_stream_completed")

    async def test_cleanup_removes_generation_and_calls_all_resources(self):
        publisher = MagicMock()
        publisher.stop = AsyncMock()
        stream_server.active_rtvi_sessions["client"] = {
            "session_id": "generation-a",
            "publisher": publisher,
            "caption_task": None,
            "rtvi_stream_id": "stream-1",
        }
        with patch.object(
            stream_server.rtvi_client, "cleanup_stream", new=AsyncMock()
        ) as cleanup_stream:
            await stream_server._cleanup_rtvi_session("client", "test")
        self.assertNotIn("client", stream_server.active_rtvi_sessions)
        cleanup_stream.assert_awaited_once_with("stream-1")
        publisher.stop.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
