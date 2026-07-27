"""Regression tests for executable generated-tool lifecycle resources."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "tools"))

from generated_tool_runtime import (  # noqa: E402
    FrameStore,
    ToolFrame,
    ToolRuntime,
    has_executable_lifecycle,
    invoke_tool_hook,
    load_generated_tool,
)
import empty_seat_detection  # noqa: E402
import validate_generated_tools  # noqa: E402


class TestGeneratedToolRuntime(unittest.IsolatedAsyncioTestCase):
    def runtime(self, emitted, *, tool="tool", session="session", accepted=True):
        async def emit(event):
            emitted.append(event)
            return accepted

        return ToolRuntime(
            client_id="client",
            tool_name=tool,
            tool_version="version",
            session_id=session,
            request_id=f"request-{session}",
            frame_store=FrameStore(max_frames=4),
            emit_callback=emit,
        )

    async def test_generated_lifecycle_hooks_are_executed(self):
        source = """
async def on_stream_start(runtime, input_data):
    runtime.set_state("started", input_data["value"])

async def on_frame(runtime, frame):
    await runtime.emit(str(frame.frame_id), partial=True)

async def on_stream_stop(runtime):
    runtime.set_state("stopped", True)
"""
        self.assertTrue(has_executable_lifecycle(source))
        namespace = load_generated_tool(source, filename="generated.py")
        emitted = []
        runtime = self.runtime(emitted)
        await invoke_tool_hook(namespace, "on_stream_start", runtime, {"value": 7})
        await invoke_tool_hook(
            namespace,
            "on_frame",
            runtime,
            ToolFrame(3, 1.0, np.zeros((2, 2, 3)), 2, 2),
        )
        await invoke_tool_hook(namespace, "on_stream_stop", runtime)

        self.assertEqual(runtime.get_state("started"), 7)
        self.assertTrue(runtime.get_state("stopped"))
        self.assertEqual(emitted[0]["text"], "3")

    async def test_recent_frames_are_bounded_and_filterable(self):
        runtime = self.runtime([])
        for frame_id in range(1, 7):
            runtime._frames.add(
                ToolFrame(frame_id, float(frame_id), object(), 10, 20)
            )
        self.assertEqual(
            [frame.frame_id for frame in runtime.get_recent_frames()], [3, 4, 5, 6]
        )
        self.assertEqual(
            [frame.frame_id for frame in runtime.get_recent_frames(count=2)], [5, 6]
        )
        self.assertEqual(
            [frame.frame_id for frame in runtime.get_recent_frames(seconds=1.5)],
            [5, 6],
        )

    async def test_one_tool_can_emit_multiple_results(self):
        emitted = []
        runtime = self.runtime(emitted)
        self.assertTrue(await runtime.emit("first", partial=True))
        self.assertTrue(await runtime.emit("second", partial=True))
        self.assertTrue(await runtime.emit("", final=True))
        self.assertEqual([event["result_index"] for event in emitted], [1, 2, 3])
        self.assertTrue(emitted[-1]["final"])

    async def test_stale_output_is_rejected_by_transport_callback(self):
        emitted = []
        runtime = self.runtime(emitted, accepted=False)
        self.assertFalse(await runtime.emit("stale", final=True))
        self.assertEqual(emitted[0]["text"], "stale")

    async def test_state_is_isolated_between_tool_sessions(self):
        first = self.runtime([], tool="one", session="a")
        second = self.runtime([], tool="one", session="b")
        third = self.runtime([], tool="two", session="a")
        first.set_state("anchor", "first")
        self.assertIsNone(second.get_state("anchor"))
        self.assertIsNone(third.get_state("anchor"))

    async def test_take_photo_and_streaming_hooks_may_differ(self):
        source = """
async def on_take_photo(runtime, image, input_data):
    return "take"
async def on_frame(runtime, frame):
    await runtime.emit("stream", final=True)
"""
        namespace = load_generated_tool(source, filename="different.py")
        emitted = []
        runtime = self.runtime(emitted)
        take = await invoke_tool_hook(
            namespace, "on_take_photo", runtime, object(), {}
        )
        await invoke_tool_hook(
            namespace,
            "on_frame",
            runtime,
            ToolFrame(1, 1.0, object(), 1, 1),
        )
        self.assertEqual(take, "take")
        self.assertEqual(emitted[0]["text"], "stream")


class TestExecutableEmptySeatTool(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _to_thread_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def test_take_photo_calls_exact_selected_model(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Two empty seats."))]
        )
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch.object(
            empty_seat_detection, "call_model", return_value=response
        ) as call, patch.object(
            empty_seat_detection.asyncio,
            "to_thread",
            side_effect=self._to_thread_inline,
        ):
            result = await empty_seat_detection.on_take_photo(None, image, {})

        self.assertEqual(result, "Two empty seats.")
        self.assertEqual(
            call.call_args.args[0], "gemini/gemini-3.1-flash-lite-preview"
        )

    async def test_tool_owned_similarity_suppresses_same_scene(self):
        emitted = []

        async def emit(event):
            emitted.append(event)
            return True

        runtime = ToolRuntime(
            client_id="client",
            tool_name="empty_seat_detection",
            tool_version="v1",
            session_id="session",
            frame_store=FrameStore(),
            emit_callback=emit,
        )
        await empty_seat_detection.on_stream_start(runtime, {})
        image = np.full((32, 32, 3), 80, dtype=np.uint8)
        frame1 = ToolFrame(1, 1.0, image, 32, 32)
        frame2 = ToolFrame(2, 2.0, image.copy(), 32, 32)
        frame3 = ToolFrame(3, 3.0, image.copy(), 32, 32)

        with patch.object(
            empty_seat_detection,
            "_call_selected_model",
            side_effect=lambda model, provider, frame: f"{model} result",
        ) as calls, patch.object(
            empty_seat_detection.asyncio,
            "to_thread",
            side_effect=self._to_thread_inline,
        ):
            await empty_seat_detection.on_frame(runtime, frame1)
            await empty_seat_detection.on_frame(runtime, frame2)
            await empty_seat_detection.on_frame(runtime, frame3)

        self.assertEqual(calls.call_count, 3)
        self.assertCountEqual(
            [call.args[0] for call in calls.call_args_list],
            [model for model, _provider in empty_seat_detection.PROGRESSIVE_MODELS],
        )
        self.assertEqual(
            [call.args[0] for call in calls.call_args_list].count("gpt-5"),
            1,
        )
        self.assertEqual(len([event for event in emitted if event["partial"]]), 3)
        self.assertEqual(len([event for event in emitted if event["final"]]), 1)


class TestExecutableToolValidation(unittest.TestCase):
    def test_allows_direct_model_calls_async_and_local_similarity(self):
        source = '''
import asyncio
import numpy as np
from litellm_utils import call_model, extract_text
TOOL_NAME = "example"
TOOL_PROMPT = "Describe the requested visual information."
def similarity(a, b):
    return float(np.dot(a, b))
async def on_take_photo(runtime, image, input_data):
    response = await asyncio.to_thread(
        call_model, "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}], [image]
    )
    return extract_text(response)
'''
        failures = validate_generated_tools.validate_take_photo_tool(
            source, "issue", Path("tools/example.py")
        )
        self.assertEqual(failures, [])

    def test_rejects_environment_network_process_and_filesystem_access(self):
        cases = {
            "environment": "import os\nvalue = os.getenv('OPENAI_API_KEY')",
            "network": "import requests\nrequests.get('https://example.com')",
            "process": "import subprocess\nsubprocess.run(['echo'])",
            "filesystem": "open('/tmp/result', 'w')",
        }
        for label, body in cases.items():
            with self.subTest(label=label):
                path = validate_generated_tools.TOOLS_DIR / f"_unsafe_{label}.py"
                try:
                    path.write_text(body, encoding="utf-8")
                    failures = validate_generated_tools.validate_files([path])
                finally:
                    path.unlink(missing_ok=True)
                self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
