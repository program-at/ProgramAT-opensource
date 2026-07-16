import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litellm_utils
import stream_server
from validate_generated_tools import validate_take_photo_baseline


class TestTakePhotoBaseline(unittest.TestCase):
    @staticmethod
    def _completion(text):
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": text})()
            })()]
        })()

    def test_checked_in_user_tools_have_unified_take_photo_prompt_constants(self):
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

    def test_evaluator_yes_returns_gemini_without_gpt5(self):
        responses = [self._completion("The label says aspirin."), self._completion(" yes\n")]
        with patch.object(litellm_utils, "call_model", side_effect=responses) as call, \
             patch.object(litellm_utils, "_call_gpt5_responses") as gpt5:
            answer = litellm_utils.call_take_photo_baseline_vlm(
                image=b"image", prompt="Read the label."
            )

        self.assertEqual(answer, "The label says aspirin.")
        self.assertEqual(call.call_count, 2)
        call.assert_any_call(
            model_name=litellm_utils.TAKE_PHOTO_BASELINE_MODEL,
            messages=[{"role": "user", "content": "Read the label."}],
            images=[b"image"],
            metadata={"num_retries": 0},
        )
        evaluator_call = call.call_args_list[1].kwargs
        self.assertEqual(evaluator_call["model_name"], "openai/gpt-4o-mini")
        self.assertIn("The label says aspirin.", evaluator_call["messages"][0]["content"])
        self.assertEqual(evaluator_call["images"], [])
        gpt5.assert_not_called()

    def test_evaluator_no_calls_gpt5_with_only_original_inputs(self):
        image = object()
        responses = [self._completion("Gemini failed answer marker."), self._completion("NO")]
        with patch.object(litellm_utils, "call_model", side_effect=responses), \
             patch.object(
                 litellm_utils, "_call_gpt5_responses", return_value="GPT-5 answer"
             ) as gpt5:
            answer = litellm_utils.call_c2_no_result_passing(
                image, "Original fused prompt", mode="take-photo", request_id="photo-1"
            )

        self.assertEqual(answer, "GPT-5 answer")
        gpt5.assert_called_once_with(image, "Original fused prompt")
        fallback_args = " ".join(repr(arg) for arg in gpt5.call_args.args)
        self.assertNotIn("Gemini failed answer marker", fallback_args)

    def test_gemini_failure_directly_calls_gpt5_and_skips_evaluator(self):
        image = b"original-image"
        with patch.object(litellm_utils, "call_model", side_effect=TimeoutError("secret")) as call, \
             patch.object(
                 litellm_utils, "_call_gpt5_responses", return_value="Fallback answer"
             ) as gpt5:
            answer = litellm_utils.call_c2_no_result_passing(
                image, "Original prompt", mode="streaming", request_id="frame-7"
            )

        self.assertEqual(answer, "Fallback answer")
        self.assertEqual(call.call_count, 1)
        gpt5.assert_called_once_with(image, "Original prompt")

    def test_gpt5_responses_request_preserves_multimodal_format_and_reasoning(self):
        client = type("Client", (), {})()
        client.responses = type("Responses", (), {})()
        create = __import__("unittest.mock", fromlist=["Mock"]).Mock(
            return_value=type("Response", (), {"output_text": "Fresh fallback"})()
        )
        client.responses.create = create
        with patch("openai.OpenAI", return_value=client) as openai_client:
            answer = litellm_utils._call_gpt5_responses(
                b"raw-image", "Original fused prompt"
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

    def test_take_photo_validator_accepts_exactly_one_baseline_call(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
from litellm_utils import call_take_photo_baseline_vlm
TOOL_NAME = "read_label"
TOOL_PROMPT = "Read the label."
def main(image, input_data):
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        self.assertEqual(validate_take_photo_baseline(tool, issue, Path("tools/read_label.py")), [])

    def test_take_photo_validator_rejects_router_and_multiple_calls(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
TOOL_PROMPT = "Read the label."
TOOL_NAME = "read_label"
def main(image, input_data):
    copilot_llm_call(capability="ocr", images=[image])
    call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        failures = validate_take_photo_baseline(tool, issue, Path("tools/read_label.py"))
        self.assertTrue(any("exactly one" in failure for failure in failures))
        self.assertTrue(any("must not call copilot_llm_call" in failure for failure in failures))

    def test_simple_copilot_prompt_uses_unified_contract(self):
        prompt = (
            "Identify the visible hand gesture. Return only the gesture name; "
            "if no gesture is clear, say 'No clear gesture.'"
        )
        issue = "## Mode\n\ntake-photo\n"
        tool = f'''
from litellm_utils import call_take_photo_baseline_vlm
TOOL_NAME = "find_seat"
TOOL_PROMPT = {prompt!r}
def main(image, input_data):
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        self.assertEqual(validate_take_photo_baseline(tool, issue, Path("tools/find_seat.py")), [])
        self.assertNotIn("1.", prompt)
        self.assertNotIn("First", prompt)

        authored_call = tool.replace("prompt=TOOL_PROMPT", 'prompt="Describe the image."')
        failures = validate_take_photo_baseline(authored_call, issue, Path("tools/find_seat.py"))
        self.assertTrue(any("pass TOOL_PROMPT directly" in failure for failure in failures))

    def test_complex_copilot_prompt_fuses_ordered_subtasks_into_one_call(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
from litellm_utils import call_take_photo_baseline_vlm
TOOL_NAME = "find_uber"
TOOL_PROMPT = (
    "Follow this sequence using the same image: 1. Identify the likely rideshare vehicle "
    "using visible make, model, color, or other distinguishing features. 2. Read the "
    "license plate if visible and explicitly say when it cannot be confirmed. 3. Identify "
    "the appropriate passenger-side entry point. 4. Give concise navigation guidance to "
    "the vehicle. Return only the final user-facing answer."
)
def main(image, input_data):
    return call_take_photo_baseline_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
'''
        self.assertEqual(validate_take_photo_baseline(tool, issue, Path("tools/find_uber.py")), [])
        tree = ast.parse(tool)
        helper_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "call_take_photo_baseline_vlm"
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
            if isinstance(node, ast.FunctionDef) and node.name == "_run_take_photo_baseline"
        )
        calls = [
            node.func.id for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertEqual(calls.count("call_take_photo_baseline_vlm"), 1)
        self.assertEqual(calls.count("_take_photo_tool_prompt"), 1)
        self.assertNotIn("system_llm_call", calls)
        self.assertNotIn("copilot_llm_call", calls)
        self.assertNotIn("execute_capability_sequence", calls)

    def test_runtime_sends_copilot_prompt_to_gemini_exactly_once(self):
        code = 'TOOL_NAME = "gesture"\nTOOL_PROMPT = "Return only the visible gesture."\n'
        with patch.object(
            stream_server, "call_take_photo_baseline_vlm", return_value="Waving"
        ) as gemini, patch.object(stream_server, "tool_copilot_llm_call") as router:
            result = stream_server._run_take_photo_baseline("gesture", code, b"image")

        self.assertEqual(result, "Waving")
        gemini.assert_called_once_with(
            image=b"image", prompt="Return only the visible gesture.",
            mode="take-photo", request_id=None,
        )
        router.assert_not_called()

    def test_runtime_rejects_tools_without_a_prompt(self):
        with self.assertRaisesRegex(ValueError, "no string TOOL_PROMPT"):
            stream_server._run_take_photo_baseline("legacy", "def main(): pass", b"image")

    def test_runtime_bypass_is_connected_to_take_photo_and_streaming(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        references = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "_run_take_photo_baseline"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(references), 2)
        self.assertIn("if data.get('type') == 'run_tool':", source)
        self.assertIn("'streaming'", source)


if __name__ == "__main__":
    unittest.main()
