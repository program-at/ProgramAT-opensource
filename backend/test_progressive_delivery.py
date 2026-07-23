"""End-to-end tests for the progressive executor-to-WebSocket bridge."""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import patch

import policy_executor
import stream_server


TOOL_CODE = '''
TOOL_NAME = "progressive_test"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "Describe the image."
TOOL_POLICY = {
    "strategy": "parallel_progressive",
    "models": ["moondream", "gemini-3.1-flash-lite", "gpt-5"],
}
'''


class RecordingWebSocket:
    def __init__(self):
        self.started = time.monotonic()
        self.messages = []

    async def send(self, payload):
        self.messages.append((time.monotonic() - self.started, json.loads(payload)))


def mocked_runtime(delays):
    def execute(
        prompt, image, *, policy, request_id, metadata, on_progress,
        on_progress_error, is_cancelled,
    ):
        def call(model, _prompt, _image, _metadata):
            time.sleep(delays[model])
            return f"{model} result"

        runtime_metadata = dict(metadata)
        runtime_metadata.update({
            "request_id": request_id,
            "progressive_model_timeout_seconds": float(
                os.environ.get("PROGRESSIVE_MODEL_TIMEOUT_SECONDS", "60")
            ),
        })
        return policy_executor.execute_tool_policy(
            policy, prompt, image, call, metadata=runtime_metadata,
            on_progress=on_progress, on_progress_error=on_progress_error,
            is_cancelled=is_cancelled,
        )

    return execute


class ProgressiveDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        stream_server.active_progressive_invocations.clear()

    async def run_invocation(self, delays):
        websocket = RecordingWebSocket()
        with patch.object(
            stream_server, "execute_resolved_tool_policy",
            side_effect=mocked_runtime(delays),
        ):
            await stream_server._start_progressive_invocation(
                websocket, "client", "progressive_test", TOOL_CODE,
                b"image", "take-photo",
            )
            record = stream_server.active_progressive_invocations[
                ("client", "progressive_test")
            ]
            await record["task"]
        return websocket.messages

    async def test_results_are_sent_at_model_completion_times(self):
        messages = await self.run_invocation({
            "moondream": 0.1,
            "gemini-3.1-flash-lite": 0.3,
            "gpt-5": 1.0,
        })
        results = [
            (elapsed, message) for elapsed, message in messages
            if message["type"] == "tool_progress_result" and not message["final"]
        ]
        final = [
            (elapsed, message) for elapsed, message in messages
            if message["type"] == "tool_progress_result" and message["final"]
        ]

        self.assertEqual(
            [message["model"] for _, message in results],
            ["moondream", "gemini-3.1-flash-lite", "gpt-5"],
        )
        self.assertEqual([message["result_index"] for _, message in results], [1, 2, 3])
        self.assertLess(results[0][0], 0.3)
        self.assertLess(results[1][0], 1.0)
        self.assertGreaterEqual(results[2][0], 0.9)
        self.assertEqual(len(final), 1)
        self.assertGreaterEqual(final[0][0], results[2][0])

    async def test_timed_out_model_does_not_prevent_final_marker(self):
        with patch.dict(os.environ, {"PROGRESSIVE_MODEL_TIMEOUT_SECONDS": "0.15"}):
            messages = await self.run_invocation({
                "moondream": 0.02,
                "gemini-3.1-flash-lite": 0.04,
                "gpt-5": 0.5,
            })
        results = [
            message for _, message in messages
            if message["type"] == "tool_progress_result" and not message["final"]
        ]
        self.assertEqual([message["model"] for message in results], [
            "moondream", "gemini-3.1-flash-lite", "gpt-5",
        ])
        self.assertIn("timed out", results[-1]["error"])
        self.assertTrue(messages[-1][1]["final"])
        self.assertLess(messages[-1][0], 0.4)


if __name__ == "__main__":
    unittest.main()
