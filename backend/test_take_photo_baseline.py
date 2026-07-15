import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litellm_utils
from validate_generated_tools import validate_take_photo_baseline


class TestTakePhotoBaseline(unittest.TestCase):
    def test_helper_makes_one_fixed_model_call_without_retries(self):
        response = type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": "The label says aspirin."})()
            })()]
        })()
        with patch.object(litellm_utils, "call_model", return_value=response) as call:
            answer = litellm_utils.call_take_photo_baseline_vlm(
                image=b"image", prompt="Read the label."
            )

        self.assertEqual(answer, "The label says aspirin.")
        call.assert_called_once_with(
            model_name=litellm_utils.TAKE_PHOTO_BASELINE_MODEL,
            messages=[{"role": "user", "content": "Read the label."}],
            images=[b"image"],
            metadata={"num_retries": 0},
        )

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

    def test_p1_prompt_is_mechanical_and_must_be_copied_verbatim(self):
        prompt = (
            "Inspect the image for seats, determine which are unoccupied, select the nearest "
            "available seat, and return only concise spoken directions."
        )
        issue = f"## Mode\n\ntake-photo\n\n## Prompt strategy\n\nno_planner\n\n## P1 exact prompt\n\n{prompt}\n"
        tool = f'''
from litellm_utils import call_take_photo_baseline_vlm
TOOL_NAME = "find_seat"
TOOL_PROMPT = {prompt!r}
def main(image, input_data):
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        self.assertEqual(validate_take_photo_baseline(tool, issue, Path("tools/find_seat.py")), [])

        rewritten = tool.replace("concise spoken directions", "directions")
        failures = validate_take_photo_baseline(rewritten, issue, Path("tools/find_seat.py"))
        self.assertTrue(any("exactly match" in failure for failure in failures))

        authored_call = tool.replace("prompt=TOOL_PROMPT", 'prompt="Describe the image."')
        failures = validate_take_photo_baseline(authored_call, issue, Path("tools/find_seat.py"))
        self.assertTrue(any("pass TOOL_PROMPT directly" in failure for failure in failures))

    def test_copilot_fused_prompt_may_be_authored_in_one_tool_prompt(self):
        issue = """## Mode

take-photo

## Prompt strategy

copilot_fused_prompt

## P1 exact prompt

"""
        tool = '''
from litellm_utils import call_take_photo_baseline_vlm
TOOL_NAME = "find_seat"
TOOL_PROMPT = (
    "Inspect the image for seating, determine which seats are unoccupied, select the "
    "nearest suitable option, and return only concise spoken guidance toward it. "
    "If none is visible, say so."
)
def main(image, input_data):
    return call_take_photo_baseline_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
'''
        self.assertEqual(validate_take_photo_baseline(tool, issue, Path("tools/find_seat.py")), [])

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

    def test_runtime_bypass_is_only_in_run_tool_branch(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        references = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "_run_take_photo_baseline"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(references), 1)
        self.assertIn("if data.get('type') == 'run_tool':", source)


if __name__ == "__main__":
    unittest.main()
