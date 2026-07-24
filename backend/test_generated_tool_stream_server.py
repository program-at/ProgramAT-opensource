"""Integration coverage for executable hooks in the active server paths."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np

import stream_server
import litellm_utils


STREAM_TOOL = '''
TOOL_NAME = "integration_tool"
EXECUTION_MODE = "take_photo"

async def on_stream_start(runtime, input_data):
    runtime.set_state("started", True)

async def on_frame(runtime, frame):
    await runtime.emit(
        f"frame {frame.frame_id}", partial=True,
        metadata={"width": frame.width, "model": "explicit/test-model"},
    )

async def on_stream_stop(runtime):
    runtime.set_state("stopped", True)
'''

TAKE_PHOTO_TOOL = '''
TOOL_NAME = "take_tool"
EXECUTION_MODE = "take_photo"

async def on_take_photo(runtime, image, input_data):
    return f"selected:{input_data['value']}:{image.shape[1]}"
'''

DIRECT_MODEL_TOOL = '''
from litellm_utils import call_model
TOOL_NAME = "direct_model"
EXECUTION_MODE = "take_photo"

async def on_take_photo(runtime, image, input_data):
    return call_model(
        "provider/explicit-model",
        [{"role": "user", "content": "Use exactly this model."}],
        [image],
    )
'''


class TestGeneratedToolServerIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        configs = list(stream_server.active_streaming_tools.values())
        stream_server.active_streaming_tools.clear()
        for config in configs:
            await stream_server._stop_executable_streaming_tool(config)

    async def test_active_streaming_path_calls_hooks_and_emits(self):
        websocket = AsyncMock()
        config = {
            "tool": {
                "name": "integration_tool",
                "code": STREAM_TOOL,
                "input": {"value": 1},
            }
        }
        stream_server.active_streaming_tools["client"] = config

        await stream_server._initialize_executable_streaming_tool(
            websocket, "client", config
        )
        self.assertTrue(config["generated_runtime"].get_state("started"))

        image = np.zeros((12, 20, 3), dtype=np.uint8)
        await stream_server._dispatch_executable_tool_frame(
            "client", config, image, "base64", 10.0
        )
        await asyncio.gather(*list(config["generated_tasks"]))

        payloads = [json.loads(call.args[0]) for call in websocket.send.await_args_list]
        result = next(payload for payload in payloads if payload["type"] == "tool_progress_result")
        self.assertEqual(result["text"], "frame 1")
        self.assertEqual(result["metadata"]["width"], 20)
        self.assertEqual(result["metadata"]["model"], "explicit/test-model")
        self.assertEqual(result["model"], "explicit/test-model")

    async def test_replaced_stream_rejects_stale_emit(self):
        websocket = AsyncMock()
        config = {
            "tool": {
                "name": "integration_tool",
                "code": STREAM_TOOL,
                "input": {},
            }
        }
        stream_server.active_streaming_tools["client"] = config
        await stream_server._initialize_executable_streaming_tool(
            websocket, "client", config
        )
        stream_server.active_streaming_tools["client"] = {"replacement": True}

        accepted = await config["generated_runtime"].emit("stale", final=True)

        self.assertFalse(accepted)
        websocket.send.assert_not_awaited()

    async def test_stop_cancels_frame_tasks_and_rejects_late_emit(self):
        websocket = AsyncMock()
        config = {
            "tool": {
                "name": "integration_tool",
                "code": STREAM_TOOL,
                "input": {},
            }
        }
        stream_server.active_streaming_tools["client"] = config
        await stream_server._initialize_executable_streaming_tool(
            websocket, "client", config
        )
        pending = asyncio.create_task(asyncio.Event().wait())
        config["generated_tasks"].add(pending)

        stream_server.active_streaming_tools.pop("client")
        await stream_server._stop_executable_streaming_tool(config)
        accepted = await config["generated_runtime"].emit("late", final=True)

        self.assertTrue(config["cancelled"])
        self.assertTrue(pending.cancelled())
        self.assertFalse(accepted)
        websocket.send.assert_not_awaited()

    async def test_active_take_photo_path_executes_hook(self):
        websocket = AsyncMock()
        image = np.zeros((5, 9, 3), dtype=np.uint8)

        result, emitted = await stream_server._run_executable_take_photo(
            websocket,
            "client",
            "take_tool",
            TAKE_PHOTO_TOOL,
            image,
            "",
            {"value": "explicit"},
        )

        self.assertEqual(result, "selected:explicit:9")
        self.assertEqual(emitted, 0)
        websocket.send.assert_not_awaited()

    async def test_backend_does_not_replace_explicit_model(self):
        websocket = AsyncMock()
        image = np.zeros((5, 9, 3), dtype=np.uint8)
        with patch.object(litellm_utils, "call_model", return_value="explicit") as call:
            result, emitted = await stream_server._run_executable_take_photo(
                websocket,
                "client",
                "direct_model",
                DIRECT_MODEL_TOOL,
                image,
                "",
                {},
            )

        self.assertEqual(result, "explicit")
        self.assertEqual(emitted, 0)
        self.assertEqual(call.call_args.args[0], "provider/explicit-model")


if __name__ == "__main__":
    unittest.main()
