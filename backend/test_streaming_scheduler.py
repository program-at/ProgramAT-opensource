"""Offline regression tests for streaming key-frame single-flight scheduling."""

import asyncio
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stream_server


class TestMobileResponseBoundary(unittest.TestCase):
    def test_extracts_response_from_dict_and_execution_object(self):
        expected = "Turn slightly right and walk forward to reach the exit."
        structured = {
            "response": expected,
            "artifact": {"detections": [{"label": "exit"}]},
            "implementation": "gpt4o",
            "capability": "navigation",
            "metadata": {"latency_ms": 42},
        }
        execution_result = SimpleNamespace(response=expected, artifact=structured["artifact"])

        self.assertEqual(stream_server._response_field_only(structured), expected)
        self.assertEqual(stream_server._response_field_only(execution_result), expected)
        self.assertEqual(stream_server._response_field_only(expected), expected)

    def test_mobile_payload_contains_only_response_text(self):
        expected = "Turn slightly right and walk forward to reach the exit."
        structured = {
            "response": expected,
            "artifact": {"detections": [{"label": "exit"}]},
            "implementation": "gpt4o",
            "capability": "navigation",
            "metadata": {"latency_ms": 42},
            "audio": {
                "type": "speech",
                "text": "internal object leaked here",
                "metadata": {"internal": True},
            },
        }

        with self.assertLogs(stream_server.logger, level="DEBUG"):
            payload = stream_server._build_mobile_tool_response(
                "tool_result", "exit_tool", structured, datetime(2026, 7, 1), "debug print"
            )

        self.assertEqual(payload["result"], expected)
        self.assertEqual(payload["audio"]["text"], expected)
        self.assertNotIn("artifact", payload)
        self.assertNotIn("implementation", payload)
        self.assertNotIn("capability", payload)
        self.assertNotIn("metadata", payload)
        self.assertNotIn("metadata", payload["audio"])
        self.assertNotIn("debug print", payload["result"])

    def test_unwraps_python_repr_and_json_execution_results(self):
        expected = "Walk straight ahead toward the wooden door. The door handle is located at 2 o'clock."
        structured = {
            "response": expected,
            "artifact": {"text": expected},
            "implementation": "gemini",
            "capability": "navigation",
        }

        for serialized in (repr(structured), json.dumps(structured)):
            payload = stream_server._build_mobile_tool_response(
                "tool_result", "locate_nearest_exit", serialized, datetime(2026, 7, 1)
            )
            self.assertEqual(payload["result"], expected)
            self.assertEqual(payload["audio"]["text"], expected)
            self.assertNotIn("artifact", payload["result"])

    def test_unknown_object_is_not_stringified_into_mobile_payload(self):
        opaque = SimpleNamespace(artifact={"secret": "internal"})
        payload = stream_server._build_mobile_tool_response(
            "tool_stream_result", "tool", opaque, datetime(2026, 7, 1)
        )

        self.assertEqual(payload["result"], "Tool executed (no output)")
        self.assertEqual(payload["audio"]["text"], "Tool executed (no output)")
        self.assertNotIn("internal", payload["result"])

    def test_final_spoken_log_contains_only_response_text(self):
        expected = "The door is at 2 o'clock."
        payload = {
            "result": expected,
            "audio": {"text": expected},
            "debug": {"artifact": "internal"},
        }
        with self.assertLogs(stream_server.logger, level="INFO") as logs:
            stream_server._log_final_tool_response("door_tool", payload)

        info_logs = "\n".join(logs.output)
        self.assertIn("[FINAL SPOKEN RESPONSE]\n" + expected, info_logs)
        self.assertNotIn("artifact", info_logs)


class TestStreamingScheduler(unittest.IsolatedAsyncioTestCase):
    selector_config = {
        "implementation": "clip",
        "model": "openai/clip-vit-base-patch32",
        "similarity_threshold": 0.985,
        "max_skip_frames": 20,
    }

    @staticmethod
    async def _to_thread_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def asyncSetUp(self):
        self.interval_patcher = patch.object(
            stream_server, "MIN_STREAMING_EXECUTION_INTERVAL", 0.0
        )
        self.interval_patcher.start()

    async def asyncTearDown(self):
        tasks = [
            config.get("cascade_task")
            for config in stream_server.active_streaming_tools.values()
            if config.get("cascade_task") is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        stream_server.active_streaming_tools.clear()
        self.interval_patcher.stop()

    async def test_similar_frame_is_skipped_without_updating_embedding(self):
        previous = np.array([1.0, 0.0], dtype=np.float32)
        current = np.array([0.999, 0.001], dtype=np.float32)
        stream_server.active_streaming_tools["client"] = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "last_processed_embedding": previous,
        }
        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", return_value=current), \
             patch.object(stream_server, "run_streaming_tools") as run, \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server.schedule_streaming_frame(None, "client", "image", "base64")
            await asyncio.sleep(0)

        run.assert_not_called()
        self.assertIn("[Streaming] similarity=1.000 skip", "\n".join(logs.output))
        self.assertIs(
            stream_server.active_streaming_tools["client"]["last_processed_embedding"],
            previous,
        )

    async def test_single_flight_keeps_only_newest_pending_frame(self):
        stream_server.active_streaming_tools["client"] = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
        }
        calls = []
        embeddings = {
            "b64-1": np.array([1.0, 0.0], dtype=np.float32),
            "b64-2": np.array([0.0, 1.0], dtype=np.float32),
            "b64-3": np.array([-1.0, 0.0], dtype=np.float32),
        }

        async def run(_websocket, _client_id, image, _image_base64):
            calls.append(image)
            if image == "frame-1":
                await asyncio.sleep(0.1)

        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", side_effect=lambda frame, _: embeddings[frame]), \
             patch.object(stream_server, "run_streaming_tools", side_effect=run):
            await stream_server.schedule_streaming_frame(None, "client", "frame-1", "b64-1")
            await asyncio.sleep(0.01)
            await stream_server.schedule_streaming_frame(None, "client", "frame-2", "b64-2")
            await stream_server.schedule_streaming_frame(None, "client", "frame-3", "b64-3")
            task = stream_server.active_streaming_tools["client"]["cascade_task"]
            await asyncio.wait_for(task, 1)

        self.assertEqual(calls, ["frame-1", "frame-3"])
        self.assertNotIn("pending_key_frame", stream_server.active_streaming_tools["client"])

    async def test_execution_lock_prevents_overlapping_direct_runs(self):
        stream_server.active_streaming_tools["client"] = {
            "tool": {"name": "tool"},
            "execution_lock": asyncio.Lock(),
        }
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def execute(*_args):
            calls.append("run")
            started.set()
            await release.wait()

        with patch.object(stream_server, "_execute_streaming_tools_unlocked", side_effect=execute), \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            first = asyncio.create_task(
                stream_server.run_streaming_tools(None, "client", "frame-1", "b64-1")
            )
            await started.wait()
            await stream_server.run_streaming_tools(None, "client", "frame-2", "b64-2")
            release.set()
            await first

        self.assertEqual(calls, ["run"])
        output = "\n".join(logs.output)
        self.assertIn("[Streaming] skipped frame (already running)", output)
        self.assertIn("[Streaming] execution finished", output)

    def test_token_embeddings_are_mean_pooled_and_normalized(self):
        embedding = np.zeros((1, 50, 768), dtype=np.float32)
        embedding[:, :, 0] = 2.0
        embedding[:, :, 1] = 1.0

        vector = stream_server._normalize_streaming_embedding(embedding)

        self.assertEqual(vector.shape, (768,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        self.assertGreater(vector[0], vector[1])

    def test_structured_clip_output_uses_pooled_embedding(self):
        import torch

        pooled = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        output = SimpleNamespace(pooler_output=pooled)
        encoder = stream_server.ClipFrameEncoder("test-clip")
        encoder._processor = lambda **_kwargs: {}
        encoder._model = SimpleNamespace(
            get_image_features=lambda **_kwargs: output
        )

        with patch.object(encoder, "_load"), patch.object(encoder, "_image", return_value=object()):
            vector = encoder.encode("frame")

        np.testing.assert_allclose(vector, [0.6, 0.8], atol=1e-6)

    async def test_minimum_interval_runs_newest_pending_frame(self):
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "min_execution_interval": 0.05,
        }
        stream_server.active_streaming_tools["client"] = tool_config
        calls = []
        first_finished = asyncio.Event()
        embeddings = {
            f"b64-{index}": np.array([float(index), 1.0], dtype=np.float32)
            for index in range(1, 5)
        }

        async def run(_websocket, _client_id, image, _image_base64):
            calls.append(image)
            if image == "frame-1":
                await asyncio.sleep(0.01)
                first_finished.set()
            tool_config["last_execution_completed_at"] = asyncio.get_running_loop().time()

        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", side_effect=lambda frame, _: embeddings[frame]), \
             patch.object(stream_server, "run_streaming_tools", side_effect=run), \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server.schedule_streaming_frame(None, "client", "frame-1", "b64-1")
            await asyncio.sleep(0)
            await stream_server.schedule_streaming_frame(None, "client", "frame-2", "b64-2")
            await stream_server.schedule_streaming_frame(None, "client", "frame-3", "b64-3")
            await first_finished.wait()
            await stream_server.schedule_streaming_frame(None, "client", "frame-4", "b64-4")
            await asyncio.wait_for(tool_config["cascade_task"], 1)

        self.assertEqual(calls, ["frame-1", "frame-4"])
        output = "\n".join(logs.output)
        self.assertIn("minimum interval active", output)
        self.assertIn("pending frame replaced", output)

    async def test_max_skip_frames_forces_periodic_processing(self):
        config = {**self.selector_config, "max_skip_frames": 2}
        stream_server.active_streaming_tools["client"] = {
            "tool": {"name": "tool"},
            "frame_selector_config": config,
        }
        calls = []

        async def run(_websocket, _client_id, image, _image_base64):
            calls.append(image)

        embedding = np.array([1.0, 0.0], dtype=np.float32)
        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", return_value=embedding), \
             patch.object(stream_server, "run_streaming_tools", side_effect=run):
            for index in range(1, 5):
                await stream_server.schedule_streaming_frame(
                    None, "client", f"frame-{index}", f"b64-{index}"
                )
                task = stream_server.active_streaming_tools["client"].get("cascade_task")
                if task is not None:
                    await task

        self.assertEqual(calls, ["frame-1", "frame-4"])

    def test_selector_config_loads_from_execution_policy(self):
        config = stream_server.load_streaming_frame_selector_config()
        self.assertEqual(config, self.selector_config)


if __name__ == "__main__":
    unittest.main()
