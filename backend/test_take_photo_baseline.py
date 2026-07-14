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
TOOL_PROMPT = "Read the label."
def main(image, input_data):
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
'''
        self.assertEqual(validate_take_photo_baseline(tool, issue, Path("tools/read_label.py")), [])

    def test_take_photo_validator_rejects_router_and_multiple_calls(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
TOOL_PROMPT = "Read the label."
def main(image, input_data):
    copilot_llm_call(capability="ocr", images=[image])
    call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
'''
        failures = validate_take_photo_baseline(tool, issue, Path("tools/read_label.py"))
        self.assertTrue(any("exactly one" in failure for failure in failures))
        self.assertTrue(any("must not call copilot_llm_call" in failure for failure in failures))

    def test_runtime_bypass_is_only_in_run_tool_branch(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        references = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "_run_take_photo_no_planner"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(references), 1)
        self.assertIn("if data.get('type') == 'run_tool':", source)


if __name__ == "__main__":
    unittest.main()
