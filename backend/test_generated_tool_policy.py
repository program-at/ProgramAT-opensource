import unittest
from pathlib import Path

from validate_generated_tools import FORBIDDEN_PATTERNS, validate_take_photo_tool


def tool(policy_block: str = "", policy_argument: str = "") -> str:
    return f'''from model_execution import execute_tool_policy
TOOL_NAME = "test"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = "Describe the image."
{policy_block}
def main(image, input_data):
    return execute_tool_policy(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME{policy_argument})
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

    def test_rejects_legacy_helper_and_missing_policy(self):
        source = tool().replace(
            "from model_execution import execute_tool_policy",
            "from litellm_utils import call_take_photo_vlm",
        ).replace("execute_tool_policy(", "call_take_photo_vlm(")
        failures = self.validate(source)
        self.assertTrue(any("must not use call_take_photo_vlm" in failure for failure in failures))
        self.assertTrue(any("require a literal TOOL_POLICY" in failure for failure in failures))

    def test_rejects_custom_provider_calls_and_routing_loops(self):
        direct_provider = tool(
            'TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}',
            ", policy=TOOL_POLICY",
        ) + "\nfrom openai import OpenAI\n"
        self.assertTrue(any(
            label == "direct provider SDK import" and pattern.search(direct_provider)
            for label, pattern in FORBIDDEN_PATTERNS
        ))

        looped = tool(
            'TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}',
            ", policy=TOOL_POLICY",
        ).replace(
            "return execute_tool_policy(",
            "for attempt in range(2):\n        return execute_tool_policy(",
        )
        self.assertTrue(any("custom routing loop" in failure for failure in self.validate(looped)))


if __name__ == "__main__":
    unittest.main()
