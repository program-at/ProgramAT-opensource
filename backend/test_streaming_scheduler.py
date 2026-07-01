"""Offline regression tests for streaming key-frame single-flight scheduling."""

import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stream_server


class TestStreamingScheduler(unittest.IsolatedAsyncioTestCase):
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

    async def test_no_discards_frame_without_running_cascade(self):
        stream_server.active_streaming_tools["client"] = {"tool": {"name": "tool"}}
        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "streaming_key_frame_decision", return_value=False), \
             patch.object(stream_server, "run_streaming_tools") as run:
            await stream_server.schedule_streaming_frame(None, "client", "image", "base64")
            await asyncio.sleep(0)

        run.assert_not_called()

    async def test_single_flight_keeps_only_newest_pending_frame(self):
        stream_server.active_streaming_tools["client"] = {"tool": {"name": "tool"}}
        calls = []

        async def run(_websocket, _client_id, image, _image_base64):
            calls.append(image)
            if image == "frame-1":
                await asyncio.sleep(0.1)

        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "streaming_key_frame_decision", return_value=True), \
            patch.object(stream_server, "run_streaming_tools", side_effect=run):
            await stream_server.schedule_streaming_frame(None, "client", "frame-1", "b64-1")
            await asyncio.sleep(0.01)
            await stream_server.schedule_streaming_frame(None, "client", "frame-2", "b64-2")
            await stream_server.schedule_streaming_frame(None, "client", "frame-3", "b64-3")
            task = stream_server.active_streaming_tools["client"]["cascade_task"]
            await asyncio.wait_for(task, 1)

        self.assertEqual(calls, ["frame-1", "frame-3"])
        self.assertNotIn("pending_key_frame", stream_server.active_streaming_tools["client"])


if __name__ == "__main__":
    unittest.main()
