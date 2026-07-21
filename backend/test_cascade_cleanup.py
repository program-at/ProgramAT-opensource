import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import litellm_utils
import model_router
import stream_server


class TestCascadeCleanup(unittest.TestCase):
    @staticmethod
    def _completion(text):
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": text})()
            })()]
        })()

    def test_complex_fused_prompt_keeps_ordered_steps_in_one_runtime_prompt(self):
        prompt = (
            "Follow this sequence using the same image:\n"
            "1. Identify visible seating.\n"
            "2. Determine which seats are empty.\n"
            "3. Give concise guidance to the nearest empty seat.\n"
            "Return only the final guidance."
        )
        code = f'TOOL_NAME = "seat_finder"\nEXECUTION_MODE = "take_photo"\nTOOL_PROMPT = {prompt!r}\n'
        with patch.object(
            stream_server, "call_take_photo_vlm", return_value="Go left."
        ) as shared:
            answer = stream_server._run_take_photo_vlm(
                "seat_finder", code, b"image"
            )

        self.assertEqual(answer, "Go left.")
        shared.assert_called_once_with(
            image=b"image", prompt=prompt, mode="take-photo", request_id=None
        )

    def test_temporal_take_photo_contract_has_clear_error(self):
        code = 'EXECUTION_MODE = "hosted_video_streaming"\n'
        self.assertEqual(
            stream_server._take_photo_mode_error(code),
            'This tool requires recent visual history and is only supported in streaming mode.',
        )
        self.assertIsNone(stream_server._take_photo_mode_error(
            'EXECUTION_MODE = "take_photo"\n'
        ))

    def test_both_modes_resolve_the_same_gemini_gpt5_c3_policy(self):
        policies = [
            model_router.resolve_execution_policy(mode)
            for mode in ("take-photo", "streaming")
        ]
        for policy in policies:
            self.assertEqual(policy.candidates, ("gemini_flash_lite", "gpt5"))
            self.assertEqual(policy.evaluator, "gpt4o-mini")
            self.assertEqual(policy.condition, "C3_PASS_FAILED_ATTEMPTS")
            self.assertEqual(policy.planner_mode, "FUSED_PROMPT")

    def test_rejected_gemini_answer_is_passed_to_gpt5_with_original_inputs(self):
        policy = model_router.resolve_execution_policy("take-photo")
        image = object()
        candidate_calls = {}

        def executor(profile, messages, images, metadata):
            if profile.name == policy.evaluator:
                return model_router.ImplementationResult(self._completion("NO"))
            candidate_calls[profile.name] = (messages, images)
            answer = "Gemini failed answer" if profile.name == "gemini_flash_lite" else "GPT-5 answer"
            return model_router.ImplementationResult(self._completion(answer))

        executors = {profile.kind: executor for profile in policy.implementations.values()}
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            answer = model_router.run_cascade(policy, "Original TOOL_PROMPT", image)

        self.assertEqual(answer, "GPT-5 answer")
        gpt5_messages, gpt5_images = candidate_calls["gpt5"]
        self.assertEqual(gpt5_images, [image])
        self.assertIn("Original TOOL_PROMPT", gpt5_messages[0]["content"])
        self.assertIn("Failed attempt 1:\nGemini failed answer", gpt5_messages[0]["content"])

    def test_policy_file_alone_changes_active_cascade(self):
        source = yaml.safe_load(Path(model_router.EXECUTION_POLICY_PATH).read_text())
        source["cascade_profiles"]["default_reasoning"]["candidates"] = ["gpt5"]
        source["cascade_profiles"]["default_reasoning"]["evaluator"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution_policy.yaml"
            path.write_text(yaml.safe_dump(source))
            policy = model_router.resolve_execution_policy("streaming", path)
        self.assertEqual(policy.candidates, ("gpt5",))
        self.assertEqual(policy.evaluator, "")

    def test_no_runtime_planner_or_obsolete_experiment_configuration_remains(self):
        backend = Path(__file__).resolve().parent
        self.assertFalse((backend / "stage_decomposition.py").exists())
        policy_text = (backend / "execution_policy.yaml").read_text()
        for obsolete in (
            "planner_enabled", "routing_enabled", "difficulty_start",
            "parallel_models", "aggregator", "C1", "C2", "C4", "C5",
        ):
            self.assertNotIn(obsolete, policy_text)
        policy = yaml.safe_load(policy_text)
        self.assertEqual(set(policy), {"global", "streaming", "mode_execution", "implementations", "cascade_profiles"})
        self.assertEqual(set(policy["cascade_profiles"]), {"default_reasoning"})

    def test_shared_helper_is_used_for_take_photo_and_streaming(self):
        with patch.object(
            model_router, "run_mode_cascade", side_effect=("photo", "stream")
        ) as shared:
            photo = litellm_utils.call_take_photo_vlm(
                b"photo", "Photo prompt", mode="take-photo"
            )
            stream = litellm_utils.call_take_photo_vlm(
                b"frame", "Stream prompt", mode="streaming"
            )
        self.assertEqual((photo, stream), ("photo", "stream"))
        self.assertEqual(
            [call.kwargs["mode"] for call in shared.call_args_list],
            ["take-photo", "streaming"],
        )


if __name__ == "__main__":
    unittest.main()
