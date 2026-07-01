"""Offline regression tests for streaming key-frame single-flight scheduling."""

import asyncio
import numpy as np
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stream_server


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

    async def asyncTearDown(self):
        tasks = [
            config.get("cascade_task")
            for config in stream_server.active_streaming_tools.values()
            if config.get("cascade_task") is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        stream_server.active_streaming_tools.clear()

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
             patch.object(stream_server, "run_streaming_tools") as run:
            await stream_server.schedule_streaming_frame(None, "client", "image", "base64")
            await asyncio.sleep(0)

        run.assert_not_called()
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
