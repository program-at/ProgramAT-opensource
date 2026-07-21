import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litellm_utils
import model_router
import stream_server
from validate_generated_tools import validate_take_photo_tool


class TestTakePhotoVlm(unittest.TestCase):
    @staticmethod
    def _completion(text):
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": text})()
            })()]
        })()

    def test_every_checked_in_assistive_tool_has_one_task_specific_prompt(self):
        tools_dir = Path(__file__).resolve().parent.parent / "tools"
        tool_names = (
            "camera_aiming", "clothing_recognition", "door_detection",
            "empty_seat_detection", "live_ocr", "object_recognition",
            "scene_description",
        )
        for tool_name in tool_names:
            tree = ast.parse((tools_dir / f"{tool_name}.py").read_text(encoding="utf-8"))
            constants = {
                node.targets[0].id: node.value.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            }
            self.assertEqual(constants["TOOL_NAME"], tool_name)
            self.assertTrue(constants["TOOL_PROMPT"].strip())

    def test_evaluator_receives_exact_untruncated_tool_prompt_and_candidate(self):
        policy = model_router.resolve_execution_policy("take-photo")
        prompt = (
            "Identify important mail and briefly label each important item. "
            + "Preserve this task-specific detail. " * 200
            + "END_OF_EXACT_TOOL_PROMPT"
        )
        candidate = '{"mail_type":"Medical/Healthcare","confidence":0.98}'
        evaluator_messages = []
        decisions = iter(("NO", "YES"))

        def executor(profile, messages, images, metadata):
            if profile.name == policy.evaluator:
                evaluator_messages.append(messages)
                text = next(decisions)
            elif profile.name == "gemini_flash_lite":
                text = candidate
            else:
                text = "Important: medical letter."
            return model_router.ImplementationResult(self._completion(text), {"text": text})

        executors = {
            profile.kind: executor for profile in policy.implementations.values()
        }
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            answer = model_router.run_cascade(policy, prompt, b"image", request_id="mail-1")

        self.assertEqual(answer, "Important: medical letter.")
        first_evaluation = evaluator_messages[0]
        self.assertEqual(first_evaluation[0]["role"], "system")
        self.assertIn("actually satisfies", first_evaluation[0]["content"])
        self.assertIn("requested output format", first_evaluation[0]["content"])
        self.assertIn("prefer NO so the cascade can try the next model", first_evaluation[0]["content"])
        self.assertIn("only one specific detail cannot be confirmed", first_evaluation[0]["content"])
        self.assertEqual(first_evaluation[1]["role"], "user")
        evaluator_input = first_evaluation[1]["content"]
        self.assertIn(f"<task_prompt>\n{prompt}\n</task_prompt>", evaluator_input)
        self.assertIn(f"<candidate_answer>\n{candidate}\n</candidate_answer>", evaluator_input)
        self.assertIn("END_OF_EXACT_TOOL_PROMPT", evaluator_input)
        self.assertIn("input_truncated=false", "\n".join(logs.output))

    def test_changing_tool_prompt_changes_evaluator_outcome_without_code_change(self):
        policy = model_router.resolve_execution_policy("take-photo")

        def run(prompt):
            order = []

            def executor(profile, messages, images, metadata):
                order.append(profile.name)
                if profile.name == policy.evaluator:
                    task_and_answer = messages[1]["content"]
                    text = "YES" if "Return JSON" in task_and_answer else "NO"
                else:
                    text = f"answer-from-{profile.name}"
                return model_router.ImplementationResult(
                    self._completion(text), {"text": text}
                )

            executors = {
                profile.kind: executor for profile in policy.implementations.values()
            }
            with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
                answer = model_router.run_cascade(policy, prompt, b"image")
            return answer, order

        plain_answer, plain_order = run(
            "Identify important mail and briefly label each important item."
        )
        json_answer, json_order = run(
            "Identify important mail. Return JSON with a mail_type field."
        )

        self.assertEqual(plain_answer, "answer-from-gpt5")
        self.assertEqual(json_answer, "answer-from-gemini_flash_lite")
        self.assertEqual(plain_order[-1], "gpt5")
        self.assertEqual(json_order, ["gemini_flash_lite", "gpt4o-mini"])

    def test_no_visible_content_advances_without_accepting_first_candidate(self):
        policy = model_router.resolve_execution_policy("take-photo")
        order = []

        def executor(profile, messages, images, metadata):
            order.append(profile.name)
            if profile.name == "gemini_flash_lite":
                text = "No car is clearly visible."
            elif profile.name == policy.evaluator:
                text = "YES"
            else:
                text = "A black Toyota sedan is directly ahead; the plate is unreadable."
            return model_router.ImplementationResult(self._completion(text), {"text": text})

        executors = {
            profile.kind: executor for profile in policy.implementations.values()
        }
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            answer = model_router.run_cascade(
                policy,
                "Identify the car. If none is visible, say 'No car is clearly visible.'",
                b"image",
            )

        self.assertEqual(
            answer,
            "A black Toyota sedan is directly ahead; the plate is unreadable.",
        )
        self.assertEqual(order, ["gemini_flash_lite", "gpt5"])
        self.assertIn(
            "reason=no_relevant_information_sufficiency_rule",
            "\n".join(logs.output),
        )

    def test_useful_partial_answer_still_reaches_evaluator(self):
        policy = model_router.resolve_execution_policy("take-photo")
        order = []
        partial = "The car appears to be a black Toyota, but the license plate is not readable."

        def executor(profile, messages, images, metadata):
            order.append(profile.name)
            text = "YES" if profile.name == policy.evaluator else partial
            return model_router.ImplementationResult(self._completion(text), {"text": text})

        executors = {
            profile.kind: executor for profile in policy.implementations.values()
        }
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            answer = model_router.run_cascade(policy, "Identify my Uber.", b"image")

        self.assertEqual(answer, partial)
        self.assertEqual(order, ["gemini_flash_lite", "gpt4o-mini"])

    def test_both_modes_resolve_order_and_evaluator_from_execution_policy(self):
        for mode in ("take-photo", "streaming"):
            policy = model_router.resolve_execution_policy(mode)
            self.assertEqual(
                policy.candidates,
                ("gemini_flash_lite", "gpt5"),
            )
            self.assertEqual(policy.evaluator, "gpt4o-mini")
            self.assertEqual(policy.result_passing, "failed_attempts")
            self.assertEqual(policy.condition, "C3_PASS_FAILED_ATTEMPTS")
            self.assertEqual(policy.planner_mode, "FUSED_PROMPT")

    def test_take_photo_and_streaming_use_same_shared_policy_executor(self):
        with patch.object(
            model_router, "run_mode_cascade", side_effect=("photo", "stream")
        ) as shared:
            photo = litellm_utils.call_take_photo_vlm(
                b"photo", "Photo prompt", mode="take-photo", request_id="p1"
            )
            stream = litellm_utils.call_take_photo_vlm(
                b"frame", "Streaming prompt", mode="streaming", request_id="s1"
            )

        self.assertEqual((photo, stream), ("photo", "stream"))
        self.assertEqual(shared.call_count, 2)
        self.assertEqual(
            [call.kwargs["mode"] for call in shared.call_args_list],
            ["take-photo", "streaming"],
        )
        self.assertEqual(
            [call.kwargs["prompt"] for call in shared.call_args_list],
            ["Photo prompt", "Streaming prompt"],
        )

    def test_policy_executor_passes_ordered_failures_with_original_inputs_in_c3(self):
        policy = model_router.resolve_execution_policy("take-photo")
        image = object()
        order = []
        candidate_calls = []
        decisions = iter(("NO evaluator-secret-1", "NO evaluator-secret-2"))

        def executor(profile, messages, images, metadata):
            order.append(profile.name)
            if profile.name == policy.evaluator:
                return model_router.ImplementationResult(
                    self._completion(next(decisions)), {"text": "decision"}
                )
            candidate_calls.append((profile.name, messages, images, metadata))
            return model_router.ImplementationResult(
                self._completion(f"answer-from-{profile.name}"),
                {"text": f"answer-from-{profile.name}"},
            )

        executors = {
            profile.kind: executor for profile in policy.implementations.values()
        }
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            with self.assertLogs(model_router.logger, level="INFO") as captured_logs:
                answer = model_router.run_cascade(
                    policy, "Original fused prompt", image, request_id="photo-1"
                )

        self.assertEqual(answer, "answer-from-gpt5")
        self.assertEqual(
            order,
            ["gemini_flash_lite", "gpt4o-mini", "gpt5"],
        )
        self.assertEqual([call[0] for call in candidate_calls], list(policy.candidates))
        first_prompt = candidate_calls[0][1][0]["content"]
        second_prompt = candidate_calls[1][1][0]["content"]
        self.assertEqual(first_prompt, "Original fused prompt")
        self.assertIn("Original fused prompt", second_prompt)
        self.assertIn("Failed attempt 1:\nanswer-from-gemini_flash_lite", second_prompt)
        for index, (_name, messages, images, metadata) in enumerate(candidate_calls):
            self.assertEqual(len(messages), 1)
            self.assertEqual(len(images), 1)
            self.assertIs(images[0], image)
            self.assertTrue(metadata["preserve_original_prompt"])
            self.assertEqual(metadata["task_text"], "Original fused prompt")
            self.assertEqual(metadata["failed_attempt_count"], index)
            serialized = repr((messages, images, metadata))
            self.assertNotIn("evaluator-secret", serialized)

        handoff_logs = [
            line for line in captured_logs.output
            if "[Policy Cascade C3 Handoff]" in line
        ]
        self.assertEqual(len(handoff_logs), 1)
        self.assertIn("target_candidate=gpt5", handoff_logs[0])
        self.assertIn("failed_attempt_count=1", handoff_logs[0])
        self.assertIn("'source_model': 'gemini_flash_lite'", handoff_logs[0])
        self.assertIn("'preview': 'answer-from-gemini_flash_lite'", handoff_logs[0])
        for line in handoff_logs:
            self.assertIn("original_image_reused=true", line)
            self.assertIn("evaluator_feedback_passed=false", line)
            self.assertNotIn("evaluator-secret", line)

    def test_selected_answer_is_not_rewritten_or_reformatted(self):
        policy = model_router.resolve_execution_policy("take-photo")
        raw_answer = "**Vehicle**\n- Black Toyota\n- Plate uncertain\n{\"status\":\"partial\"}"
        evaluator_inputs = []

        def executor(profile, messages, images, metadata):
            if profile.name == policy.evaluator:
                evaluator_inputs.append(messages[1]["content"])
                text = "YES"
            else:
                text = f"  {raw_answer}  "
            return model_router.ImplementationResult(self._completion(text), {"text": text})

        executors = {
            profile.kind: executor for profile in policy.implementations.values()
        }
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            answer = model_router.run_cascade(policy, "Identify the vehicle.", b"image")

        self.assertEqual(answer, raw_answer)
        self.assertIn(f"<candidate_answer>\n{raw_answer}\n</candidate_answer>", evaluator_inputs[0])

    def test_streaming_uses_the_same_c3_failed_attempt_context(self):
        policy = model_router.resolve_execution_policy("streaming")
        image = object()
        candidate_prompts = []

        def executor(profile, messages, images, metadata):
            if profile.name == policy.evaluator:
                text = "NO" if len(candidate_prompts) == 1 else "YES"
            else:
                candidate_prompts.append(messages[0]["content"])
                self.assertIs(images[0], image)
                text = f"stream-answer-{profile.name}"
            return model_router.ImplementationResult(self._completion(text), {"text": text})

        executors = {
            profile.kind: executor for profile in policy.implementations.values()
        }
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            with self.assertLogs(model_router.logger, level="INFO") as captured_logs:
                answer = model_router.run_cascade(
                    policy, "Original streaming fused prompt", image
                )

        self.assertEqual(answer, "stream-answer-gpt5")
        self.assertEqual(candidate_prompts[0], "Original streaming fused prompt")
        self.assertIn("Original streaming fused prompt", candidate_prompts[1])
        self.assertIn(
            "Failed attempt 1:\nstream-answer-gemini_flash_lite",
            candidate_prompts[1],
        )
        handoff_log = next(
            line for line in captured_logs.output
            if "[Policy Cascade C3 Handoff]" in line
        )
        self.assertIn("mode=streaming", handoff_log)
        self.assertIn("target_candidate=gpt5", handoff_log)
        self.assertIn("failed_attempt_count=1", handoff_log)
        self.assertIn("'source_model': 'gemini_flash_lite'", handoff_log)
        self.assertIn("original_image_reused=true", handoff_log)
        self.assertIn("evaluator_feedback_passed=false", handoff_log)

    def test_evaluator_yes_stops_policy_cascade_before_later_models(self):
        policy = model_router.resolve_execution_policy("streaming")
        order = []

        def executor(profile, messages, images, metadata):
            order.append(profile.name)
            text = "YES" if profile.name == policy.evaluator else "Useful Moondream answer"
            return model_router.ImplementationResult(self._completion(text), {"text": text})

        executors = {
            profile.kind: executor for profile in policy.implementations.values()
        }
        with patch.dict(model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True):
            answer = model_router.run_cascade(policy, "Prompt", b"frame")

        self.assertEqual(answer, "Useful Moondream answer")
        self.assertEqual(order, ["gemini_flash_lite", "gpt4o-mini"])

    def test_changing_yaml_candidate_list_changes_actual_mode_cascade(self):
        source = Path(model_router.EXECUTION_POLICY_PATH)
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["cascade_profiles"]["default_reasoning"]["candidates"] = [
            "gemini_flash_lite", "gpt5"
        ]
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "execution_policy.yaml"
            policy_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            policies = {
                mode: model_router.resolve_execution_policy(mode, policy_path)
                for mode in ("take-photo", "streaming")
            }

        for mode, policy in policies.items():
            with self.subTest(mode=mode):
                order = []

                def executor(profile, messages, images, metadata):
                    order.append(profile.name)
                    text = "NO" if profile.name == policy.evaluator else f"{profile.name}-answer"
                    return model_router.ImplementationResult(
                        self._completion(text), {"text": text}
                    )

                executors = {
                    profile.kind: executor for profile in policy.implementations.values()
                }
                with patch.dict(
                    model_router.IMPLEMENTATION_EXECUTORS, executors, clear=True
                ):
                    answer = model_router.run_cascade(policy, "Prompt", b"image")

                self.assertEqual(answer, "gpt5-answer")
                self.assertEqual(
                    order, ["gemini_flash_lite", "gpt4o-mini", "gpt5"]
                )

    def test_gpt5_responses_request_preserves_multimodal_format_and_reasoning(self):
        client = type("Client", (), {})()
        client.responses = type("Responses", (), {})()
        create = __import__("unittest.mock", fromlist=["Mock"]).Mock(
            return_value=type("Response", (), {"output_text": "Fresh fallback"})()
        )
        client.responses.create = create
        with patch("openai.OpenAI", return_value=client) as openai_client:
            answer = litellm_utils.call_openai_responses_model(
                "gpt-5", "Original fused prompt", b"raw-image"
            )

        self.assertEqual(answer, "Fresh fallback")
        openai_client.assert_called_once_with(api_key=litellm_utils.resolve_api_key("gpt-5"), max_retries=0)
        request = create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        content = request["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "Original fused prompt"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_take_photo_validator_accepts_exactly_one_vlm_call(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
from litellm_utils import call_take_photo_vlm
TOOL_NAME = "read_label"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "Read the label."
def main(image, input_data):
    return call_take_photo_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        self.assertEqual(validate_take_photo_tool(tool, issue, Path("tools/read_label.py")), [])

    def test_take_photo_validator_rejects_router_and_multiple_calls(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
TOOL_PROMPT = "Read the label."
TOOL_NAME = "read_label"
EXECUTION_MODE = "take_photo"
def main(image, input_data):
    copilot_llm_call(capability="ocr", images=[image])
    call_take_photo_vlm(image=image, prompt=TOOL_PROMPT)
    return call_take_photo_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        failures = validate_take_photo_tool(tool, issue, Path("tools/read_label.py"))
        self.assertTrue(any("exactly one" in failure for failure in failures))
        self.assertTrue(any("must not call copilot_llm_call" in failure for failure in failures))

    def test_simple_copilot_prompt_uses_unified_contract(self):
        prompt = (
            "Identify the visible hand gesture. Return only the gesture name; "
            "if no gesture is clear, say 'No clear gesture.'"
        )
        issue = "## Mode\n\ntake-photo\n"
        tool = f'''
from litellm_utils import call_take_photo_vlm
TOOL_NAME = "find_seat"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = {prompt!r}
def main(image, input_data):
    return call_take_photo_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        self.assertEqual(validate_take_photo_tool(tool, issue, Path("tools/find_seat.py")), [])
        self.assertNotIn("1.", prompt)
        self.assertNotIn("First", prompt)

        authored_call = tool.replace("prompt=TOOL_PROMPT", 'prompt="Describe the image."')
        failures = validate_take_photo_tool(authored_call, issue, Path("tools/find_seat.py"))
        self.assertTrue(any("pass TOOL_PROMPT directly" in failure for failure in failures))

    def test_complex_copilot_prompt_fuses_ordered_subtasks_into_one_call(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
from litellm_utils import call_take_photo_vlm
TOOL_NAME = "find_uber"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Follow this sequence using the same image: 1. Identify the likely rideshare vehicle "
    "using visible make, model, color, or other distinguishing features. 2. Read the "
    "license plate if visible and explicitly say when it cannot be confirmed. 3. Identify "
    "the appropriate passenger-side entry point. 4. Give concise navigation guidance to "
    "the vehicle. Return only the final user-facing answer."
)
def main(image, input_data):
    return call_take_photo_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
'''
        self.assertEqual(validate_take_photo_tool(tool, issue, Path("tools/find_uber.py")), [])
        tree = ast.parse(tool)
        helper_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "call_take_photo_vlm"
        ]
        self.assertEqual(len(helper_calls), 1)
        prompt = next(
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "TOOL_PROMPT"
        )
        for marker in ("1.", "2.", "3.", "4."):
            self.assertIn(marker, prompt)

    def test_shared_runtime_has_one_helper_call_and_no_planner_or_router_calls(self):
        source_path = Path(__file__).resolve().parent / "stream_server.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_take_photo_vlm"
        )
        calls = [
            node.func.id for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertEqual(calls.count("call_take_photo_vlm"), 1)
        self.assertEqual(calls.count("_take_photo_tool_prompt"), 1)
        self.assertNotIn("system_llm_call", calls)
        self.assertNotIn("copilot_llm_call", calls)
        self.assertNotIn("execute_capability_sequence", calls)

    def test_runtime_sends_copilot_prompt_to_gemini_exactly_once(self):
        code = 'TOOL_NAME = "gesture"\nTOOL_PROMPT = "Return only the visible gesture."\n'
        with patch.object(
            stream_server, "call_take_photo_vlm", return_value="Waving"
        ) as gemini, patch.object(stream_server, "tool_copilot_llm_call") as router:
            result = stream_server._run_take_photo_vlm("gesture", code, b"image")

        self.assertEqual(result, "Waving")
        gemini.assert_called_once_with(
            image=b"image", prompt="Return only the visible gesture.",
            mode="take-photo", request_id=None,
        )
        router.assert_not_called()

    def test_runtime_uses_updated_tool_prompt_from_modified_code(self):
        original = 'TOOL_NAME = "mail"\nTOOL_PROMPT = "Identify important mail."\n'
        updated = (
            'TOOL_NAME = "mail"\n'
            'TOOL_PROMPT = "Identify important mail and briefly label each important item."\n'
        )
        with patch.object(
            stream_server, "call_take_photo_vlm", return_value="answer"
        ) as helper:
            stream_server._run_take_photo_vlm("mail", original, b"image")
            stream_server._run_take_photo_vlm("mail", updated, b"image")

        self.assertEqual(
            [call.kwargs["prompt"] for call in helper.call_args_list],
            [
                "Identify important mail.",
                "Identify important mail and briefly label each important item.",
            ],
        )

    def test_runtime_rejects_tools_without_a_prompt(self):
        with self.assertRaisesRegex(ValueError, "no string TOOL_PROMPT"):
            stream_server._run_take_photo_vlm("legacy", "def main(): pass", b"image")

    def test_runtime_bypass_is_connected_to_take_photo_and_streaming(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        references = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "_run_take_photo_vlm"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(references), 2)
        self.assertIn("if data.get('type') == 'run_tool':", source)
        self.assertIn("'streaming'", source)


if __name__ == "__main__":
    unittest.main()
