import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_router
import stream_server


class TestC5ParallelAggregation(unittest.TestCase):
    @staticmethod
    def _result(text):
        return model_router.ImplementationResult(
            type("Response", (), {
                "choices": [type("Choice", (), {
                    "message": type("Message", (), {"content": text})()
                })()]
            })()
        )

    def test_models_start_concurrently_and_aggregation_waits_for_all(self):
        policy = model_router.resolve_execution_policy("take-photo")
        barrier = threading.Barrier(3)
        finished = set()
        lock = threading.Lock()
        candidate_prompts = {}
        aggregator_observation = {}

        def executor(profile, messages, images, metadata):
            if metadata.get("aggregator"):
                self.assertEqual(profile.name, "gpt4o-mini")
                self.assertEqual(profile.kind, "model")
                with lock:
                    aggregator_observation["finished"] = set(finished)
                aggregator_observation["prompt"] = messages[0]["content"]
                aggregator_observation["images"] = list(images)
                return self._result("One concise final answer.")
            candidate_prompts[profile.name] = messages[0]["content"]
            self.assertEqual(list(images), [b"frame"])
            barrier.wait(timeout=1)
            time.sleep({"nvidia_hosted_vision": .03, "gemini_flash_lite": .02}.get(profile.name, .01))
            with lock:
                finished.add(profile.name)
            return self._result(f"answer from {profile.name}")

        executors = {profile.kind: executor for profile in policy.implementations.values()}
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            answer = model_router.run_cascade(policy, "Original task prompt", b"frame")

        self.assertEqual(answer, "One concise final answer.")
        self.assertEqual(set(candidate_prompts), set(policy.candidates))
        self.assertEqual(set(candidate_prompts.values()), {"Original task prompt"})
        self.assertEqual(aggregator_observation["finished"], set(policy.candidates))
        self.assertEqual(aggregator_observation["images"], [])
        for model in policy.candidates:
            self.assertIn(f"<{model}_answer>", aggregator_observation["prompt"])
        self.assertIn("<original_task_prompt>\nOriginal task prompt", aggregator_observation["prompt"])

    def test_partial_failures_are_aggregated_and_all_fail_is_error(self):
        policy = model_router.resolve_execution_policy("streaming")

        def partial_executor(profile, messages, images, metadata):
            if metadata.get("aggregator"):
                self.assertIn("gemini_flash_lite_answer", messages[0]["content"])
                self.assertNotIn("nvidia_hosted_vision_answer", messages[0]["content"])
                return self._result("Available answer.")
            if profile.name != "gemini_flash_lite":
                raise RuntimeError("provider failed")
            return self._result("Gemini succeeded")

        executors = {profile.kind: partial_executor for profile in policy.implementations.values()}
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            self.assertEqual(
                model_router.run_cascade(policy, "Prompt", b"frame"),
                "Available answer.",
            )

        def failed_executor(profile, messages, images, metadata):
            raise RuntimeError("provider failed")

        with patch.dict(
            model_router.IMPLEMENTATION_EXECUTORS,
            {profile.kind: failed_executor for profile in policy.implementations.values()},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "all failed"):
                model_router.run_cascade(policy, "Prompt", b"frame")

    def test_gpt4o_mini_aggregator_uses_normal_text_only_model_call(self):
        policy = model_router.resolve_execution_policy("take-photo")
        visual_finished = set()
        calls = []
        lock = threading.Lock()

        def visual_executor(profile, messages, images, metadata):
            self.assertEqual(list(images), [b"frame"])
            with lock:
                visual_finished.add(profile.name)
            return self._result(f"{profile.name} visual answer")

        def call_model(model, messages, images=None, metadata=None):
            calls.append((model, messages, list(images or []), dict(metadata or {})))
            if model == "openai/gpt-4o-mini":
                self.assertEqual(visual_finished, set(policy.candidates))
                self.assertEqual(list(images or []), [])
                return self._result("Text-only GPT-4o-mini aggregation.").response
            self.assertEqual(model, "gemini/gemini-3.1-flash-lite-preview")
            self.assertEqual(list(images or []), [b"frame"])
            with lock:
                visual_finished.add("gemini_flash_lite")
            return self._result("Gemini visual answer").response

        with patch.dict(
            model_router.IMPLEMENTATION_EXECUTORS,
            {
                "nvidia_hosted": visual_executor,
                "openai_responses": visual_executor,
                "model": model_router._model_executor,
            },
            clear=True,
        ), patch.object(model_router, "call_model", side_effect=call_model):
            answer = model_router.run_cascade(
                policy, "Original task-specific prompt", b"frame"
            )

        self.assertEqual(answer, "Text-only GPT-4o-mini aggregation.")
        self.assertEqual(calls[-1][0], "openai/gpt-4o-mini")
        self.assertEqual(calls[-1][2], [])
        aggregator_prompt = calls[-1][1][0]["content"]
        self.assertIn("Original task-specific prompt", aggregator_prompt)
        for model in policy.candidates:
            self.assertIn(f"<{model}_answer>", aggregator_prompt)

    def test_c5_skips_evaluator_and_difficulty_prediction_for_both_modes(self):
        for mode in ("take-photo", "streaming"):
            with self.subTest(mode=mode):
                policy = model_router.resolve_execution_policy(mode)
                self.assertEqual(policy.condition, "C5_PARALLEL_AGGREGATION")
                self.assertEqual(policy.evaluator, "")
                self.assertEqual(policy.difficulty_starts, {})
                self.assertEqual(policy.aggregator, "gpt4o-mini")

        code = 'TOOL_NAME = "reader"\nTOOL_PROMPT = "Read the visible text."\n'
        with patch.object(
            stream_server, "call_take_photo_baseline_vlm", side_effect=("photo", "stream")
        ) as shared, patch.object(
            stream_server, "_resolve_tool_difficulty_start"
        ) as predictor:
            photo = stream_server._run_take_photo_baseline("reader", code, b"photo")
            stream = stream_server._run_take_photo_baseline(
                "reader", code, b"frame", mode="streaming"
            )

        self.assertEqual((photo, stream), ("photo", "stream"))
        predictor.assert_not_called()
        self.assertEqual(
            [call.kwargs["mode"] for call in shared.call_args_list],
            ["take-photo", "streaming"],
        )
        self.assertTrue(all("difficulty_start" not in call.kwargs for call in shared.call_args_list))

    def test_logs_parallel_outputs_and_aggregation_input_and_output(self):
        policy = model_router.resolve_execution_policy("take-photo")

        def executor(profile, messages, images, metadata):
            if metadata.get("aggregator"):
                return self._result("Combined spoken answer")
            return self._result(f"Raw answer from {profile.name}")

        executors = {profile.kind: executor for profile in policy.implementations.values()}
        with patch.dict(
            model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True
        ), self.assertLogs(model_router.logger, level="INFO") as captured:
            model_router.run_cascade(policy, "Original task", b"frame")

        logs = "\n".join(captured.output)
        for model in policy.candidates:
            self.assertIn(f"model={model}", logs)
            self.assertIn(f"Raw answer from {model}", logs)
        self.assertIn("call_type=aggregator_call", logs)
        self.assertIn("<original_task_prompt>\\nOriginal task", logs)
        self.assertIn("output='Combined spoken answer'", logs)


if __name__ == "__main__":
    unittest.main()
