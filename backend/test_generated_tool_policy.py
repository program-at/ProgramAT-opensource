import unittest
from pathlib import Path

from validate_generated_tools import validate_take_photo_tool


def tool(policy_block: str = "", policy_argument: str = "") -> str:
    return f'''from litellm_utils import call_take_photo_vlm
TOOL_NAME = "test"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "Describe the image."
{policy_block}
def main(image, input_data):
    return call_take_photo_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME{policy_argument})
'''


class GeneratedToolPolicyValidationTests(unittest.TestCase):
    def validate(self, source: str):
        return validate_take_photo_tool(source, "", Path("tools/test.py"))

    def test_accepts_supported_literal_policy(self):
        failures = self.validate(tool(
            'TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}',
            ", policy=TOOL_POLICY",
        ))
        self.assertEqual(failures, [])

    def test_rejects_unsupported_strategy_model_and_field(self):
        cases = [
            '{"strategy": "race", "models": ["gemini-3.1-flash-lite"]}',
            '{"strategy": "single", "models": ["unknown"]}',
            '{"strategy": "single", "models": ["gemini-3.1-flash-lite"], "route": "custom"}',
        ]
        for policy in cases:
            with self.subTest(policy=policy):
                failures = self.validate(tool(f"TOOL_POLICY = {policy}", ", policy=TOOL_POLICY"))
                self.assertTrue(any("invalid TOOL_POLICY" in failure for failure in failures))

    def test_rejects_dynamic_policy_and_custom_orchestration(self):
        dynamic = tool('TOOL_POLICY = build_policy()', ", policy=TOOL_POLICY")
        self.assertTrue(any("static literal" in failure for failure in self.validate(dynamic)))


if __name__ == "__main__":
    unittest.main()
