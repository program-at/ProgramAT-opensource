"""Offline regression tests for streaming key-frame single-flight scheduling."""

import asyncio
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stream_server


class TestDisabledPlannerToolPrompt(unittest.TestCase):
    def test_resolves_declared_tool_prompt(self):
        tool_code = '''
TOOL_PROMPT = "Identify the medicine label and read the dosage."

def main(image):
    return copilot_llm_call(
        capability="ocr",
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        images=[image],
    )
'''
        self.assertEqual(
            stream_server._resolve_tool_prompt(tool_code),
            "Identify the medicine label and read the dosage.",
        )

    def test_disabled_planner_passes_tool_prompt_through_copilot_call(self):
        config = {
            "planner_enabled": False,
            "routing_enabled": False,
            "default_llm_when_routing_disabled": "gemini_flash_lite",
        }
        tool_code = 'TOOL_PROMPT = "Read only the nearest street sign."'
        expected = {"response": "Main Street"}
        with patch.object(stream_server, "load_global_execution_config", return_value=config), \
             patch.object(stream_server, "tool_copilot_llm_call", return_value=expected) as call, \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            result = stream_server._single_stage_tool_result(
                "street_sign", tool_code, "Describe the scene", b"image"
            )

        self.assertEqual(result, expected)
        self.assertEqual(
            call.call_args.kwargs["messages"],
            [{"role": "user", "content": "Read only the nearest street sign."}],
        )
        output = "\n".join(logs.output)
        self.assertIn("planner_enabled=false routing_enabled=false", output)
        self.assertIn("selected_model=gemini_flash_lite", output)
        self.assertIn("prompt_source=tool_prompt", output)


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

    def test_string_model_output_is_not_decoded_or_rewritten(self):
        structured = {
            "response": "Walk straight ahead toward the wooden door.",
            "artifact": {"text": "internal"},
            "implementation": "gemini",
            "capability": "navigation",
        }

        for serialized in (repr(structured), json.dumps(structured)):
            payload = stream_server._build_mobile_tool_response(
                "tool_result", "locate_nearest_exit", serialized, datetime(2026, 7, 1)
            )
            self.assertEqual(payload["result"], serialized)
            self.assertEqual(payload["audio"]["text"], serialized)

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
        "max_similarity_to_last_sent": 0.990,
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

    async def test_schedule_uses_clip_gating_without_debounce_timer(self):
        stream_server.active_streaming_tools["client"] = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.0,
            "candidate_max_age_seconds": 2.0,
            "min_stable_confirmations": 1,
        }
        calls = []

        async def run(_websocket, _client_id, image, _image_base64):
            calls.append(image)
            if image == "frame-1":
                await asyncio.sleep(0.1)
            return True

        embeddings = {
            "b64-1": np.array([1.0, 0.0], dtype=np.float32),
            "b64-2": np.array([0.0, 1.0], dtype=np.float32),
            "b64-3": np.array([-1.0, 0.0], dtype=np.float32),
        }

        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", side_effect=lambda frame, _: embeddings[frame]) as encode, \
             patch.object(stream_server, "run_streaming_tools", side_effect=run), \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server.schedule_streaming_frame(None, "client", "frame-1", "b64-1")
            await asyncio.sleep(0.01)
            await stream_server.schedule_streaming_frame(None, "client", "frame-2", "b64-2")
            await stream_server.schedule_streaming_frame(None, "client", "frame-3", "b64-3")
            task = stream_server.active_streaming_tools["client"]["cascade_task"]
            await asyncio.wait_for(task, 1)

        self.assertEqual(encode.call_count, 3)
        self.assertEqual(calls, ["frame-1", "frame-3"])
        self.assertIsNone(
            stream_server.active_streaming_tools["client"].get("deferred_visual_state")
        )
        output = "\n".join(logs.output)
        self.assertIn("debounce_zero=true", output)
        self.assertIn("decision=skipped_model_busy", output)

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
        self.assertIn("streaming_executor=hosted_video_streaming", output)
        self.assertIn("nvidia_hosted_active=true", output)

    async def test_c3_streaming_result_is_dropped_when_session_is_superseded(self):
        tool_config = {
            "generation": 4,
            "current_execution_id": 9,
            "tool": {
                "name": "describe",
                "language": "python",
                "code": 'TOOL_PROMPT = "Original fused streaming prompt."',
                "input": {},
            },
            "exec_env": {
                "parsed_input": {},
                "common_modules": {},
                "stdout_capture": __import__("io").StringIO(),
                "exec_globals_base": {},
            },
        }
        stream_server.active_streaming_tools["client"] = tool_config
        websocket = AsyncMock()

        def finish_after_restart(*args, **kwargs):
            self.assertEqual(args[3], "streaming")
            self.assertEqual(args[4], "9")
            stream_server.active_streaming_tools["client"] = {
                "generation": 5, "tool": tool_config["tool"]
            }
            return "Completed old-frame policy result"

        with patch.object(
            stream_server, "_run_take_photo_vlm", side_effect=finish_after_restart
        ) as cascade, patch.object(
            stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline
        ), self.assertLogs(stream_server.logger, level="INFO") as logs:
            sent = await stream_server._execute_streaming_tools_unlocked(
                websocket, "client", b"frame", "base64-frame"
            )

        self.assertFalse(sent)
        cascade.assert_called_once()
        websocket.send.assert_not_awaited()
        self.assertIn("tool policy result discarded as stale", "\n".join(logs.output))

    async def test_streaming_sends_only_final_policy_cascade_result(self):
        tool_config = {
            "generation": 4,
            "current_execution_id": 10,
            "tool": {
                "name": "describe",
                "language": "python",
                "code": 'TOOL_PROMPT = "Original fused streaming prompt."',
                "input": {},
            },
            "exec_env": {
                "parsed_input": {},
                "common_modules": {},
                "stdout_capture": __import__("io").StringIO(),
                "exec_globals_base": {},
            },
        }
        stream_server.active_streaming_tools["client"] = tool_config
        websocket = AsyncMock()

        with patch.object(
            stream_server,
            "_run_take_photo_vlm",
            return_value="Final accepted cascade answer",
        ) as cascade, patch.object(
            stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline
        ):
            sent = await stream_server._execute_streaming_tools_unlocked(
                websocket, "client", b"selected-frame", "base64-frame"
            )

        self.assertTrue(sent)
        cascade.assert_called_once_with(
            "describe",
            'TOOL_PROMPT = "Original fused streaming prompt."',
            b"selected-frame",
            "streaming",
            "10",
        )
        websocket.send.assert_awaited_once()
        payload = json.loads(websocket.send.await_args.args[0])
        self.assertEqual(payload["result"], "Final accepted cascade answer")
        self.assertEqual(payload["audio"]["text"], "Final accepted cascade answer")
        self.assertEqual(payload["execution_id"], 10)

    async def test_stable_candidate_too_similar_to_last_sent_is_skipped(self):
        now = asyncio.get_running_loop().time()
        embedding = np.array([1.0, 0.0], dtype=np.float32)
        state = {
            "state_id": 1,
            "sequence": 2,
            "image": "frame-2",
            "image_base64": "b64-2",
            "embedding": embedding,
            "started_at": now - 1.0,
            "last_seen_at": now,
            "confirmation_count": 2,
            "sent": False,
            "similarity_to_active_state": 0.99,
        }
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.6,
            "candidate_max_age_seconds": 2.0,
            "min_stable_confirmations": 2,
            "last_sent_embedding": embedding,
            "active_visual_state": state,
            "execution_lock": asyncio.Lock(),
        }
        stream_server.active_streaming_tools["client"] = tool_config

        with patch.object(stream_server, "run_streaming_tools") as run, \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server._try_send_visual_state(
                None, "client", tool_config, state, "active"
            )

        run.assert_not_called()
        self.assertTrue(tool_config["active_visual_state"]["sent"])
        self.assertIn(
            "decision=skipped_too_similar_to_last_sent",
            "\n".join(logs.output),
        )

    async def test_candidate_rechecks_last_sent_after_model_busy(self):
        now = asyncio.get_running_loop().time()
        candidate_embedding = np.array([1.0, 0.0], dtype=np.float32)
        state = {
            "state_id": 1,
            "sequence": 3,
            "image": "frame-3",
            "image_base64": "b64-3",
            "embedding": candidate_embedding,
            "started_at": now - 1.0,
            "last_seen_at": now,
            "confirmation_count": 2,
            "sent": False,
            "similarity_to_active_state": 0.99,
        }
        lock = asyncio.Lock()
        await lock.acquire()
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.6,
            "candidate_max_age_seconds": 2.0,
            "min_stable_confirmations": 2,
            "last_sent_embedding": np.array([0.0, 1.0], dtype=np.float32),
            "active_visual_state": state,
            "execution_lock": lock,
        }
        stream_server.active_streaming_tools["client"] = tool_config

        with patch.object(stream_server, "run_streaming_tools") as run, \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server._try_send_visual_state(
                None, "client", tool_config, state, "active"
            )
            lock.release()
            tool_config["last_sent_embedding"] = candidate_embedding
            await stream_server._try_send_visual_state(
                None, "client", tool_config, state, "active"
            )

        run.assert_not_called()
        output = "\n".join(logs.output)
        self.assertIn("decision=skipped_model_busy", output)
        self.assertIn("decision=skipped_too_similar_to_last_sent", output)
        self.assertIn("decision=skipped_model_busy", output)
        self.assertIn("decision=skipped_too_similar_to_last_sent", output)
        self.assertIsNone(tool_config.get("deferred_visual_state"))

    async def test_missing_clip_embedding_fails_open_and_sends(self):
        now = asyncio.get_running_loop().time()
        state = {
            "state_id": 1,
            "sequence": 4,
            "image": "frame-4",
            "image_base64": "b64-4",
            "embedding": None,
            "started_at": now - 1.0,
            "last_seen_at": now,
            "confirmation_count": 2,
            "sent": False,
            "similarity_to_active_state": 0.99,
        }
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.6,
            "candidate_max_age_seconds": 2.0,
            "min_stable_confirmations": 2,
            "active_visual_state": state,
            "execution_lock": asyncio.Lock(),
        }
        stream_server.active_streaming_tools["client"] = tool_config

        async def run_tool(_websocket, _client_id, _image, _image_base64):
            return True

        with patch.object(stream_server, "run_streaming_tools", side_effect=run_tool) as run, \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server._try_send_visual_state(
                None, "client", tool_config, state, "active"
            )
            await tool_config["cascade_task"]

        run.assert_called_once()
        output = "\n".join(logs.output)
        self.assertIn("[Streaming] clip_gating_failed_open", output)
        self.assertIn("decision=clip_gating_failed_open", output)

    async def test_stable_duration_without_confirmations_does_not_send(self):
        now = asyncio.get_running_loop().time()
        state = {
            "state_id": 1,
            "sequence": 5,
            "image": "frame-5",
            "image_base64": "b64-5",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "started_at": now - 1.0,
            "last_seen_at": now,
            "confirmation_count": 1,
            "sent": False,
            "similarity_to_active_state": None,
        }
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.6,
            "candidate_max_age_seconds": 2.0,
            "min_stable_confirmations": 2,
            "active_visual_state": state,
            "execution_lock": asyncio.Lock(),
        }
        stream_server.active_streaming_tools["client"] = tool_config

        with patch.object(stream_server, "run_streaming_tools") as run, \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server._try_send_visual_state(
                None, "client", tool_config, state, "active"
            )

        run.assert_not_called()
        self.assertIsNotNone(tool_config.get("active_visual_state"))
        self.assertEqual(tool_config["active_visual_state"]["sequence"], state["sequence"])
        self.assertIn(
            "decision=skipped_not_enough_confirmations",
            "\n".join(logs.output),
        )

    async def test_sequential_changed_frames_are_dispatched_with_clip_gating(self):
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.0,
            "candidate_max_age_seconds": 20.0,
            "min_stable_confirmations": 2,
        }
        stream_server.active_streaming_tools["client"] = tool_config
        calls = []

        async def run(_websocket, _client_id, image, _image_base64):
            calls.append(image)
            return True

        embeddings = {
            "b64-1": np.array([1.0, 0.0], dtype=np.float32),
            "b64-2": np.array([0.0, 1.0], dtype=np.float32),
            "b64-3": np.array([-1.0, 0.0], dtype=np.float32),
        }

        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", side_effect=lambda frame, _: embeddings[frame]) as encode, \
             patch.object(stream_server, "run_streaming_tools", side_effect=run):
            await stream_server.schedule_streaming_frame(None, "client", "frame-1", "b64-1")
            await tool_config["cascade_task"]
            await stream_server.schedule_streaming_frame(None, "client", "frame-2", "b64-2")
            second_task = tool_config.get("cascade_task")
            if second_task is not None:
                await second_task
            await stream_server.schedule_streaming_frame(None, "client", "frame-3", "b64-3")
            await tool_config["cascade_task"]

        self.assertEqual(encode.call_count, 3)
        self.assertEqual(calls, ["frame-1", "frame-2", "frame-3"])

    async def test_zero_debounce_does_not_create_timer(self):
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.0,
            "candidate_max_age_seconds": 20.0,
            "min_stable_confirmations": 2,
            "active_visual_state": None,
            "deferred_visual_state": None,
            "in_flight_state_embedding": None,
            "in_flight_state_id": None,
        }
        stream_server.active_streaming_tools["client"] = tool_config

        async def run(_websocket, _client_id, _image, _image_base64):
            return True

        embeddings = {
            "b64-1": np.array([1.0, 0.0], dtype=np.float32),
            "b64-2": np.array([0.0, 1.0], dtype=np.float32),
        }

        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", side_effect=lambda frame, _: embeddings[frame]) as encode, \
             patch.object(stream_server, "run_streaming_tools", side_effect=run), \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server.schedule_streaming_frame(None, "client", "frame-1", "b64-1")
            await tool_config["cascade_task"]
            await stream_server.schedule_streaming_frame(None, "client", "frame-2", "b64-2")
            second_task = tool_config.get("cascade_task")
            if second_task is not None:
                await second_task

        self.assertEqual(encode.call_count, 2)
        self.assertIsNone(tool_config.get("deferred_visual_state"))
        self.assertIsNone(tool_config.get("debounce_task"))
        output = "\n".join(logs.output)
        self.assertIn("debounce_zero=true", output)
        self.assertNotIn("visual_state_stable", output)

    async def test_almost_identical_frames_are_suppressed_by_clip_gating(self):
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.0,
            "candidate_max_age_seconds": 20.0,
            "min_stable_confirmations": 1,
        }
        stream_server.active_streaming_tools["client"] = tool_config
        calls = []

        async def run(_websocket, _client_id, image, _image_base64):
            calls.append(image)
            return True

        embeddings = {
            "b64-1": np.array([1.0, 0.0], dtype=np.float32),
            "b64-2": np.array([1.0, 0.001], dtype=np.float32),
        }

        with patch.object(stream_server.asyncio, "to_thread", side_effect=self._to_thread_inline), \
             patch.object(stream_server, "encode_streaming_frame", side_effect=lambda frame, _: embeddings[frame]) as encode, \
             patch.object(stream_server, "run_streaming_tools", side_effect=run), \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server.schedule_streaming_frame(None, "client", "frame-1", "b64-1")
            await tool_config["cascade_task"]
            await stream_server.schedule_streaming_frame(None, "client", "frame-2", "b64-2")
            second_task = tool_config.get("cascade_task")
            if second_task is not None:
                await second_task

        self.assertEqual(encode.call_count, 2)
        self.assertEqual(calls, ["frame-1"])
        output = "\n".join(logs.output)
        self.assertIn("decision=skipped_too_similar_to_last_sent", output)
        self.assertIn("max_similarity_to_last_sent=0.990", output)

    async def test_forced_key_frame_interval_sends_similar_frame(self):
        now = asyncio.get_running_loop().time()
        embedding = np.array([1.0, 0.0], dtype=np.float32)
        state = {
            "state_id": 1,
            "sequence": 6,
            "image": "frame-6",
            "image_base64": "b64-6",
            "embedding": embedding,
            "started_at": now,
            "last_seen_at": now,
            "confirmation_count": 1,
            "sent": False,
            "similarity_to_active_state": None,
        }
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.0,
            "candidate_max_age_seconds": 2.0,
            "forced_key_frame_interval_seconds": 5.0,
            "min_stable_confirmations": 1,
            "last_sent_embedding": embedding,
            "last_sent_at": now - 5.1,
            "active_visual_state": state,
            "execution_lock": asyncio.Lock(),
        }
        stream_server.active_streaming_tools["client"] = tool_config

        async def run_tool(_websocket, _client_id, _image, _image_base64):
            return True

        with patch.object(stream_server, "run_streaming_tools", side_effect=run_tool), \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server._try_send_visual_state(
                None, "client", tool_config, state, "active"
            )
            await tool_config["cascade_task"]

        output = "\n".join(logs.output)
        self.assertIn("decision=forced_key_frame_interval_elapsed", output)
        self.assertIn("forced_interval_seconds=5.0", output)

    async def test_deferred_state_dropped_if_active_state_moved_on_before_finish(self):
        now = asyncio.get_running_loop().time()
        deferred = {
            "state_id": 2,
            "sequence": 2,
            "image": "frame-2",
            "image_base64": "b64-2",
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "started_at": now - 1.0,
            "last_seen_at": now,
            "confirmation_count": 2,
            "sent": False,
            "similarity_to_active_state": 0.99,
        }
        active = {
            "state_id": 3,
            "sequence": 3,
            "image": "frame-3",
            "image_base64": "b64-3",
            "embedding": np.array([-1.0, 0.0], dtype=np.float32),
            "started_at": now,
            "last_seen_at": now,
            "confirmation_count": 1,
            "sent": False,
            "similarity_to_active_state": 0.0,
        }
        tool_config = {
            "tool": {"name": "tool"},
            "frame_selector_config": dict(self.selector_config),
            "stability_debounce_seconds": 0.6,
            "candidate_max_age_seconds": 2.0,
            "min_stable_confirmations": 2,
            "active_visual_state": active,
            "deferred_visual_state": deferred,
            "execution_lock": asyncio.Lock(),
        }
        stream_server.active_streaming_tools["client"] = tool_config

        with patch.object(stream_server, "run_streaming_tools") as run, \
             self.assertLogs(stream_server.logger, level="INFO") as logs:
            await stream_server._try_send_deferred_visual_state_after_in_flight(
                None, "client", tool_config
            )

        run.assert_not_called()
        self.assertIsNone(tool_config.get("deferred_visual_state"))
        self.assertIn(
            "reason=replaced_by_current_active_state",
            "\n".join(logs.output),
        )

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

    def test_selector_config_loads_from_execution_policy(self):
        config = stream_server.load_streaming_frame_selector_config()
        self.assertEqual(config, self.selector_config)

    def test_last_sent_similarity_threshold_environment_override(self):
        with patch.dict(
            stream_server.os.environ,
            {"STREAMING_MAX_SIMILARITY_TO_LAST_SENT": "0.91"},
        ):
            config = stream_server.load_streaming_frame_selector_config()

        self.assertEqual(config["max_similarity_to_last_sent"], 0.91)

    def test_clip_warmup_runs_once_and_logs_ready_latency(self):
        stream_server._clip_ready_models.clear()
        with patch.object(
            stream_server,
            "load_streaming_frame_selector_config",
            return_value=dict(self.selector_config),
        ), patch.object(
            stream_server,
            "encode_streaming_frame",
            return_value=np.array([1.0, 0.0], dtype=np.float32),
        ) as encode, self.assertLogs(stream_server.logger, level="INFO") as logs:
            stream_server.warm_streaming_frame_selector()
            stream_server.warm_streaming_frame_selector()

        encode.assert_called_once()
        warmed_image = encode.call_args.args[0]
        self.assertEqual(warmed_image.size, (32, 32))
        self.assertIn("[Streaming] CLIP ready latency_ms=", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
