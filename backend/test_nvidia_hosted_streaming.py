"""Offline tests for NVIDIA hosted multiframe helpers and client behavior."""

import asyncio
import base64
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nvidia_hosted_client import (
    HostedModel,
    NvidiaHostedClient,
    NvidiaHostedError,
    RollingFrameBuffer,
    TimedFrame,
    build_multi_image_request,
    build_video_request,
    encode_frames_to_mp4,
    encode_frames_as_mp4,
    parse_hosted_models,
    parse_structured_video_result,
    select_hosted_model,
    uniformly_sample_frames,
    validate_played_card_event,
)

import stream_server


class _FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._body = json.dumps(payload) if not isinstance(payload, str) else payload

    async def text(self):
        return self._body


class _FakeContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args):
        return False


class _FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, _url):
        return _FakeContext(self.response)


class TestHostedModelDiscovery(unittest.IsolatedAsyncioTestCase):
    async def test_client_discovers_models_from_mocked_api(self):
        client = NvidiaHostedClient("https://example.test/v1", "key")
        client._http = AsyncMock(return_value=_FakeSession(_FakeResponse(200, {
            "data": [
                {"id": "text/model"},
                {"id": "nvidia/nemotron-nano-12b-v2-vl"},
            ]
        })))
        models = await client.list_models()
        self.assertEqual([model.id for model in models], [
            "text/model", "nvidia/nemotron-nano-12b-v2-vl"
        ])
        self.assertTrue(models[1].supports_vision)
        self.assertTrue(models[1].supports_video)

    async def test_api_error_preserves_full_body(self):
        client = NvidiaHostedClient("https://example.test/v1", "key")
        body = "multi-image rejected: " + "detail" * 1000
        client._http = AsyncMock(return_value=_FakeSession(_FakeResponse(422, body)))
        with self.assertRaises(NvidiaHostedError) as raised:
            await client.list_models()
        self.assertIn(body, str(raised.exception))

    def test_selects_first_vision_model_or_validates_configured_model(self):
        models = parse_hosted_models({"data": [
            {"id": "text/model"},
            {"id": "vendor/vision-model", "input_modalities": ["text", "image"]},
        ]})
        self.assertEqual(select_hosted_model(models).id, "vendor/vision-model")
        self.assertEqual(
            select_hosted_model(models, "vendor/vision-model").id,
            "vendor/vision-model",
        )
        with self.assertRaisesRegex(NvidiaHostedError, "was not returned"):
            select_hosted_model(models, "missing/model")


class TestRollingFrameBuffer(unittest.TestCase):
    def test_played_card_json_requires_before_after_and_confidence(self):
        parsed = parse_structured_video_result(json.dumps({
            "event_detected": True,
            "before_cards": ["jack of diamonds", "nine of hearts"],
            "after_cards": ["nine of hearts"],
            "played_card": "jack of diamonds",
            "confidence": 0.9,
            "evidence": "Visible before and absent after",
        }))
        self.assertEqual(
            validate_played_card_event(parsed, 0.8),
            ("You just played the jack of diamonds.", "accepted"),
        )
        parsed["after_cards"].append("jack of diamonds")
        self.assertEqual(
            validate_played_card_event(parsed, 0.8)[1],
            "played_card_still_in_after",
        )

    def test_unchanged_readable_hand_emits_no_card_played(self):
        parsed = {
            "event_detected": False,
            "before_cards": ["Jack of Diamonds", "nine of hearts"],
            "after_cards": ["nine of hearts", "jack of diamonds"],
            "played_card": None,
            "confidence": 0.9,
            "evidence": "The same readable cards remain in the hand.",
        }
        self.assertEqual(
            validate_played_card_event(parsed, 0.8),
            (None, "stable_no_event"),
        )

    def test_uncertain_or_empty_hand_does_not_emit_no_change(self):
        for before, after, confidence in (([], [], 0.9),
                                          (["nine of hearts"], [], 0.9),
                                          (["nine of hearts"], ["nine of hearts"], 0.5)):
            accepted, _decision = validate_played_card_event({
                "event_detected": False,
                "before_cards": before,
                "after_cards": after,
                "played_card": None,
                "confidence": confidence,
                "evidence": "Uncertain",
            }, 0.8)
            self.assertIsNone(accepted)

    def test_played_card_semantic_state_requires_silent_stable_reset(self):
        session = {
            "semantic_state": "stable_no_event",
            "output_config": {"confidence_threshold": 0.8},
        }
        event = {
            "event_detected": True,
            "before_cards": ["jack of diamonds", "nine of hearts"],
            "after_cards": ["nine of hearts"],
            "played_card": "jack of diamonds",
            "confidence": 0.9,
        }
        accepted, previous, current, reason = (
            stream_server._apply_played_card_semantic_transition(session, event)
        )
        self.assertEqual(accepted, "You just played the jack of diamonds.")
        self.assertEqual((previous, current, reason), (
            "stable_no_event", "played_card:jack of diamonds",
            "new_played_card_transition",
        ))
        self.assertIsNone(
            stream_server._apply_played_card_semantic_transition(session, event)[0]
        )

        stable = {
            "event_detected": False,
            "before_cards": ["nine of hearts"],
            "after_cards": ["nine of hearts"],
            "played_card": None,
            "confidence": 0.9,
        }
        self.assertEqual(
            stream_server._apply_played_card_semantic_transition(session, stable)[3],
            "stable_reset",
        )
        self.assertIsNotNone(
            stream_server._apply_played_card_semantic_transition(session, event)[0]
        )

    @staticmethod
    def _frame(index):
        return f"data:image/jpeg;base64,{base64.b64encode(str(index).encode()).decode()}"

    def test_buffer_is_time_and_count_bounded(self):
        buffer = RollingFrameBuffer(window_seconds=2, max_frames=3)
        for index, timestamp in enumerate((0.0, 1.0, 2.0, 3.0)):
            buffer.add(self._frame(index), timestamp)
        snapshot = buffer.snapshot()
        self.assertEqual([frame.timestamp for frame in snapshot], [1.0, 2.0, 3.0])
        buffer.add(self._frame(4), 4.5)
        self.assertEqual([frame.timestamp for frame in buffer.snapshot()], [3.0, 4.5])
        buffer.clear()
        self.assertEqual(len(buffer), 0)

    def test_uniform_sampling_preserves_order_and_endpoints(self):
        frames = [TimedFrame(float(index), self._frame(index)) for index in range(10)]
        sampled = uniformly_sample_frames(frames, 4)
        self.assertEqual([frame.timestamp for frame in sampled], [0.0, 3.0, 6.0, 9.0])

    def test_multiple_image_request_has_text_first_and_ordered_images(self):
        frames = [TimedFrame(float(index), self._frame(index)) for index in range(3)]
        request = build_multi_image_request("vision/model", "Find the exit", frames, 80)
        content = request["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("earliest to latest", content[0]["text"])
        self.assertIn("Find the exit", content[0]["text"])
        self.assertEqual(
            [item["image_url"]["url"] for item in content[1:]],
            [frame.image_data_uri for frame in frames],
        )
        self.assertFalse(request["stream"])

    def test_native_video_request_uses_documented_video_url_shape(self):
        prompt = "Find moving obstacles\nUNCHANGED_PROMPT_END"
        request = build_video_request(
            "video/model", prompt, "data:video/mp4;base64,AAAA", 80
        )
        content = request["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertTrue(content[0]["text"].endswith(prompt))
        self.assertEqual(content[1], {
            "type": "video_url",
            "video_url": {"url": "data:video/mp4;base64,AAAA"},
        })

    def test_sampled_frames_can_be_cpu_encoded_as_mp4(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 48), "red").save(image_bytes, "JPEG")
        data_uri = "data:image/jpeg;base64," + base64.b64encode(
            image_bytes.getvalue()
        ).decode()
        video_uri = encode_frames_as_mp4([
            TimedFrame(1.0, data_uri),
            TimedFrame(1.2, data_uri),
        ], fps=5)
        self.assertTrue(video_uri.startswith("data:video/mp4;base64,"))
        self.assertGreater(len(video_uri), 100)

    def test_ffmpeg_mp4_command_is_h264_yuv420p_and_chronological(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 48), "blue").save(image_bytes, "JPEG")
        uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes.getvalue()).decode()
        frames = [TimedFrame(2, uri), TimedFrame(1, uri)]
        with __import__("tempfile").TemporaryDirectory() as directory, \
             patch("nvidia_hosted_client.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("nvidia_hosted_client.subprocess.run") as run:
            output = Path(directory) / "clip.mp4"
            def finalize(command, **_kwargs):
                output.write_bytes(b"mp4")
            run.side_effect = finalize
            self.assertEqual(encode_frames_to_mp4(frames, output, 4, 1280), 3)
            command = run.call_args.args[0]
            self.assertIn("libx264", command)
            self.assertIn("yuv420p", command)
            self.assertIn("4", command)


class TestHostedStreamingIntegration(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _session(buffer=None):
        return {
            "generation_token": "generation-a",
            "tool_name": "assistive-tool",
            "prompt": "Find the exit",
            "frame_buffer": buffer or RollingFrameBuffer(2, 20),
            "scheduler_task": None,
            "request_task": None,
            "model": HostedModel("vendor/video", True, True, {}),
            "input_format": "base64_mp4",
            "window_seconds": 2,
            "interval_seconds": 0.01,
            "overlap_seconds": 0,
            "output_config": {},
            "temporary_paths": set(),
            "last_emitted_events": {},
            "clip_sequence": 0,
            "latest_forwarded_clip": 0,
            "skipped_intervals": 0,
        }

    async def asyncTearDown(self):
        for client_id in list(stream_server.active_hosted_nvidia_sessions):
            await stream_server._cleanup_hosted_nvidia_session(client_id, "test_teardown")
        stream_server.active_rtvi_sessions.clear()
        stream_server.active_streaming_tools.clear()
        stream_server.active_streaming_tasks.clear()

    async def test_scheduler_allows_one_request_and_skips_busy_intervals(self):
        now = __import__("time").time()
        buffer = RollingFrameBuffer(2, 20)
        buffer.add("data:image/jpeg;base64,QQ==", now)
        session = self._session(buffer)
        stream_server.active_hosted_nvidia_sessions["client"] = session
        request_started = asyncio.Event()
        release = asyncio.Event()

        async def slow_request(*_args):
            request_started.set()
            await release.wait()

        model = HostedModel("vendor/video", True, True, {})
        with patch.object(
            stream_server.hosted_nvidia_client,
            "select_model",
            new=AsyncMock(return_value=model),
        ), patch.object(
            stream_server,
            "_run_hosted_nvidia_clip",
            side_effect=slow_request,
        ) as run_clip:
            scheduler = asyncio.create_task(
                stream_server._hosted_nvidia_scheduler("ws", "client", session)
            )
            session["scheduler_task"] = scheduler
            await asyncio.wait_for(request_started.wait(), 1)
            await asyncio.sleep(0.035)
            self.assertEqual(run_clip.await_count, 1)
            self.assertGreaterEqual(session["skipped_intervals"], 1)
            release.set()
            await asyncio.sleep(0)
            scheduler.cancel()
            await asyncio.gather(scheduler, return_exceptions=True)

    async def test_stale_generation_result_is_not_forwarded(self):
        now = __import__("time").time()
        frame = TimedFrame(now, "data:image/jpeg;base64,QQ==")
        session = self._session()
        stream_server.active_hosted_nvidia_sessions["client"] = session
        websocket = MagicMock()
        websocket.send = AsyncMock()

        async def replace_session(_request):
            self.assertEqual(
                _request["max_tokens"], stream_server.HOSTED_VIDEO_MAX_TOKENS
            )
            self.assertGreaterEqual(_request["max_tokens"], 256)
            stream_server.active_hosted_nvidia_sessions["client"] = self._session()
            return "Exit ahead"

        async def to_thread_inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch.object(
            stream_server.asyncio,
            "to_thread",
            side_effect=to_thread_inline,
        ), patch.object(
            stream_server,
            "encode_frames_to_mp4",
            return_value=100,
        ), patch.object(
            stream_server,
            "video_file_data_uri",
            return_value="data:video/mp4;base64,AAAA",
        ), patch.object(
            stream_server,
            "probe_mp4",
            return_value={"streams": [{"width": 320, "height": 240}], "format": {"duration": "2"}},
        ), patch.object(
            stream_server.hosted_nvidia_client,
            "chat_completions",
            side_effect=replace_session,
        ):
            await stream_server._run_hosted_nvidia_clip(
                websocket, "client", session, 1, [frame]
            )
        websocket.send.assert_not_awaited()

    async def test_cleanup_cancels_tasks_and_clears_buffer(self):
        buffer = RollingFrameBuffer(2, 20)
        buffer.add("data:image/jpeg;base64,QQ==", __import__("time").time())
        session = self._session(buffer)
        session["scheduler_task"] = asyncio.create_task(asyncio.sleep(60))
        session["request_task"] = asyncio.create_task(asyncio.sleep(60))
        stream_server.active_hosted_nvidia_sessions["client"] = session
        await stream_server._cleanup_hosted_nvidia_session("client", "test")
        self.assertNotIn("client", stream_server.active_hosted_nvidia_sessions)
        self.assertEqual(len(buffer), 0)
        self.assertTrue(session["scheduler_task"].cancelled())
        self.assertTrue(session["request_task"].cancelled())

    async def test_active_dispatch_ignores_nvidia_mode_and_uses_policy_scheduler(self):
        stream_server.active_hosted_nvidia_sessions["client"] = self._session()
        stream_server.active_rtvi_sessions["client"] = {"session_id": "rtvi"}
        stream_server.active_streaming_tools["client"] = {"tool": {"name": "tool"}}
        with patch.object(stream_server, "NVIDIA_STREAMING_MODE", "hosted_multiframe"), \
             patch.object(stream_server, "_offer_hosted_nvidia_frame", new=AsyncMock()) as hosted, \
             patch.object(stream_server, "_offer_rtvi_frame", new=AsyncMock()) as rtvi, \
             patch.object(stream_server, "schedule_streaming_frame", new=AsyncMock()) as original:
            task = await stream_server._dispatch_active_streaming_frame(
                "ws", "client", "image", "base64", 10.0
            )
            await task
        rtvi.assert_not_awaited()
        hosted.assert_not_awaited()
        original.assert_awaited_once_with("ws", "client", "image", "base64")


if __name__ == "__main__":
    unittest.main()
