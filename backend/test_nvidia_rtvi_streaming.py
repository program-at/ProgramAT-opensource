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
from validate_generated_tools import validate_rtvi_streaming_tool


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
    def test_missing_ffmpeg_has_actionable_error(self):
        publisher = RtspPublisher(
            "client",
            RtspPublisherConfig(
                public_base_url="rtsp://media:8554/programat",
                ffmpeg_binary="definitely-not-an-installed-ffmpeg",
            ),
        )

        async def run():
            with self.assertRaisesRegex(RuntimeError, "PROGRAMAT_FFMPEG_BINARY"):
                await publisher.start()

        asyncio.run(run())

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

    def test_played_card_tool_is_explicitly_rtvi_and_declarative(self):
        tool_path = Path(__file__).resolve().parent.parent / "tools" / "played_card_rtvi.py"
        tool_text = tool_path.read_text(encoding="utf-8")
        self.assertEqual(stream_server._tool_execution_mode(tool_text), "hosted_video_streaming")
        self.assertEqual(
            validate_rtvi_streaming_tool(
                tool_text, "## Mode\n\nhosted_video_streaming\n", tool_path
            ),
            [],
        )
        self.assertNotIn("call_take_photo_", tool_text)
        self.assertNotIn("Gemini", tool_text)

    def test_existing_static_tools_declare_single_frame_mode(self):
        tools_dir = Path(__file__).resolve().parent.parent / "tools"
        for name in (
            "scene_description", "live_ocr", "clothing_recognition",
            "door_detection", "empty_seat_detection", "object_recognition",
            "camera_aiming",
        ):
            text = (tools_dir / f"{name}.py").read_text(encoding="utf-8")
            self.assertEqual(stream_server._tool_execution_mode(text), "take_photo", name)
            self.assertIsNone(stream_server._literal_tool_metadata(text, "VIDEO_CONFIG"), name)

    async def test_start_creates_one_isolated_rtvi_session(self):
        websocket = MagicMock(send=AsyncMock())
        tool_path = Path(__file__).resolve().parent.parent / "tools" / "played_card_rtvi.py"
        tool_code = tool_path.read_text(encoding="utf-8")

        def publisher_factory(session_id, _config):
            publisher = MagicMock()
            publisher.start = AsyncMock()
            publisher.stop = AsyncMock()
            publisher.writer_task = None
            publisher.rtsp_url = f"rtsp://media/{session_id}"
            return publisher

        with patch.object(stream_server, "RtspPublisher", side_effect=publisher_factory), \
             patch.object(stream_server, "call_take_photo_vlm") as take_photo, \
             patch.object(stream_server, "_start_hosted_nvidia_session", new=AsyncMock()) as hosted:
            await stream_server._start_rtvi_session(websocket, "client-a", {
                "tool_name": "played_card_rtvi", "tool_code": tool_code,
            })
            await stream_server._start_rtvi_session(websocket, "client-b", {
                "tool_name": "played_card_rtvi", "tool_code": tool_code,
            })
        self.assertIn("client-a", stream_server.active_rtvi_sessions)
        self.assertIn("client-b", stream_server.active_rtvi_sessions)
        self.assertIsNot(
            stream_server.active_rtvi_sessions["client-a"],
            stream_server.active_rtvi_sessions["client-b"],
        )
        self.assertIn("chronological short video", stream_server.active_rtvi_sessions["client-a"]["prompt"])
        take_photo.assert_not_called()
        hosted.assert_not_awaited()

    def test_event_filter_suppresses_sentinel_and_overlapping_duplicates(self):
        session = {"last_emitted_events": {}}
        self.assertEqual(stream_server._filter_rtvi_event(session, "", 0, 1, 1), (None, "empty"))
        self.assertEqual(
            stream_server._filter_rtvi_event(session, "NO_EVENT", 0, 1, 2),
            (None, "no_event"),
        )
        result = "You just played the seven of hearts."
        self.assertEqual(stream_server._filter_rtvi_event(session, result, 2, 4, 3), (result, "accepted"))
        self.assertEqual(
            stream_server._filter_rtvi_event(session, result, 3, 5, 9),
            (None, "duplicate"),
        )
        self.assertEqual(
            stream_server._filter_rtvi_event(session, result, 20, 25, 20),
            (result, "accepted"),
        )

    async def test_disabled_flag_uses_original_scheduler(self):
        stream_server.active_streaming_tools["client"] = {"tool": {"name": "tool"}}
        with patch.object(stream_server, "NVIDIA_STREAMING_MODE", "original"), \
             patch.object(stream_server, "schedule_streaming_frame", new=AsyncMock()) as schedule:
            task = await stream_server._dispatch_active_streaming_frame(
                "ws", "client", "decoded-image", "jpeg-base64"
            )
            await task
        schedule.assert_awaited_once_with("ws", "client", "decoded-image", "jpeg-base64")

    async def test_valid_played_card_event_is_emitted_and_duplicate_id_suppressed(self):
        websocket = MagicMock()
        websocket.send = AsyncMock()
        publisher = MagicMock()
        publisher.wait_for_first_frame = AsyncMock()
        session = {
            "session_id": "generation-a",
            "tool_name": "played_card_rtvi",
            "prompt": "Detect played cards",
            "rtsp_url": "rtsp://media/programat/generation-a",
            "publisher": publisher,
            "rtvi_stream_id": None,
            "started_at": asyncio.get_running_loop().time(),
            "first_result_at": None,
            "forwarded_chunk_ids": set(),
            "last_emitted_events": {},
            "output_sequence": 0,
        }
        stream_server.active_rtvi_sessions["client"] = session

        async def captions(**_kwargs):
            yield RtviCaption("NO_EVENT", 6, 0, 2)
            yield RtviCaption("You just played the seven of hearts.", 7, 2, 4)
            yield RtviCaption("You just played the seven of hearts.", 7, 2, 4)

        with patch.object(stream_server, "NVIDIA_RTVI_MODEL", "rtvi-model"), \
             patch.object(stream_server.rtvi_client, "register_stream", new=AsyncMock(return_value="stream-1")), \
             patch.object(stream_server.rtvi_client, "stream_captions", side_effect=captions), \
             patch.object(stream_server, "_cleanup_rtvi_session", new=AsyncMock()) as cleanup:
            await stream_server._run_rtvi_caption_session(websocket, "client", session)

        self.assertEqual(websocket.send.await_count, 1)
        payload = json.loads(websocket.send.await_args.args[0])
        self.assertEqual(payload["result"], "You just played the seven of hearts.")
        self.assertEqual(payload["rtvi_chunk_id"], 7)
        cleanup.assert_awaited_once_with("client", reason="caption_stream_completed")

    async def test_cleanup_removes_generation_and_calls_all_resources(self):
        publisher = MagicMock()
        publisher.stop = AsyncMock()
        other_publisher = MagicMock()
        other_publisher.stop = AsyncMock()
        stream_server.active_rtvi_sessions["client"] = {
            "session_id": "generation-a",
            "publisher": publisher,
            "caption_task": None,
            "rtvi_stream_id": "stream-1",
        }
        stream_server.active_rtvi_sessions["other-client"] = {
            "session_id": "generation-b",
            "publisher": other_publisher,
            "caption_task": None,
            "rtvi_stream_id": None,
        }
        with patch.object(
            stream_server.rtvi_client, "cleanup_stream", new=AsyncMock()
        ) as cleanup_stream:
            await stream_server._cleanup_rtvi_session("client", "test")
        self.assertNotIn("client", stream_server.active_rtvi_sessions)
        self.assertIn("other-client", stream_server.active_rtvi_sessions)
        cleanup_stream.assert_awaited_once_with("stream-1")
        publisher.stop.assert_awaited_once()
        other_publisher.stop.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
