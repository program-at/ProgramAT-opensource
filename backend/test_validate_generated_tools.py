"""Tests for generated-tool router guardrails."""

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_generated_tools


class TestValidateGeneratedTools(unittest.TestCase):
    UBER_STAGE_ISSUE = """
## Feature Description
Uber pickup guidance

## Task Stages
Stage 1:
Capability: object_detection_localization

Stage 2:
Capability: spatial_reasoning

Stage 3:
Capability: navigation
"""

    def _write_temp_tool(self, name: str, text: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(dir=validate_generated_tools.TOOLS_DIR)
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_rejects_local_router_and_direct_provider_patterns(self):
        path = self._write_temp_tool(
            "bad_generated_tool.py",
            """
from ultralytics import YOLO

DEFAULT_MODEL = "some-provider/model"
COCO_CLASSES = ["car", "bus"]

class ModelRouter:
    pass

def detect_vehicles(image):
    model = YOLO("yolo11n.pt")
    return model(image)
""",
        )

        failures = validate_generated_tools.validate_files([path])

        self.assertTrue(any("direct ultralytics import" in failure for failure in failures))
        self.assertTrue(any("direct YOLO import" in failure for failure in failures))
        self.assertTrue(any("direct YOLO call" in failure for failure in failures))
        self.assertTrue(any("hardcoded YOLO model name" in failure for failure in failures))
        self.assertTrue(any("DEFAULT_MODEL constant" in failure for failure in failures))
        self.assertTrue(any("local ModelRouter class" in failure for failure in failures))
        self.assertTrue(any("COCO class list" in failure for failure in failures))
        self.assertTrue(any("model file reference" in failure for failure in failures))

    def test_allows_tool_policy_client_import(self):
        path = self._write_temp_tool(
            "good_generated_tool.py",
            """
from tool_policy_client import copilot_llm_call

TASK_CATEGORY = "general_reasoning"

def main(image, input_data=None):
    return copilot_llm_call(
        task_category=TASK_CATEGORY,
        messages=[{"role": "user", "content": "Describe the relevant scene details."}],
        images=[image],
        metadata={"tool_name": "good_generated_tool"},
    )
""",
        )

        self.assertEqual(validate_generated_tools.validate_files([path]), [])

    def test_allows_detector_agnostic_ocr_capability_call(self):
        path = self._write_temp_tool(
            "good_ocr_tool.py",
            '''
from tool_policy_client import copilot_llm_call

def main(image, input_data=None):
    result = copilot_llm_call(capability="ocr", images=[image])
    return result["response"]
''',
        )

        self.assertEqual(validate_generated_tools.validate_files([path]), [])

    def test_rejects_stringified_capability_result(self):
        for expression in ("str(guidance)", "repr(guidance)", 'f"{guidance}"'):
            path = self._write_temp_tool(
                "stringified_result_tool.py",
                f'''
from tool_policy_client import copilot_llm_call

def main(image, input_data=None):
    guidance = copilot_llm_call(
        capability="navigation",
        goal="Guide the user to the exit.",
        images=[image],
    )
    return {expression}
''',
            )

            failures = validate_generated_tools.validate_files([path])
            self.assertTrue(any("must not be stringified" in failure for failure in failures))

    def test_allows_returning_capability_response_field(self):
        path = self._write_temp_tool(
            "response_field_tool.py",
            '''
from tool_policy_client import copilot_llm_call

def main(image, input_data=None):
    guidance = copilot_llm_call(
        capability="navigation",
        goal="Guide the user to the exit.",
        images=[image],
    )
    return guidance["response"]
''',
        )

        self.assertEqual(validate_generated_tools.validate_files([path]), [])

    def test_allows_explicit_calls_matching_uber_stage_capabilities(self):
        path = self._write_temp_tool(
            "uber_three_stage_tool.py",
            """
from tool_policy_client import copilot_llm_call

def main(image, input_data=None):
    vehicle = copilot_llm_call(
        capability="object_detection_localization",
        goal="Locate the right vehicle.",
        images=[image],
        metadata={"tool_name": "uber_three_stage_tool", "target_labels": ["car"]},
    )
    door = copilot_llm_call(
        capability="spatial_reasoning",
        goal="Locate the passenger-side door.",
        messages=[{"role": "user", "content": f"Vehicle: {vehicle['artifact']}"}],
    )
    guidance = copilot_llm_call(
        capability="navigation",
        goal="Guide the user to the door.",
        messages=[{"role": "user", "content": f"Door: {door['artifact']}"}],
    )
    return guidance["response"]
""",
        )

        failures = validate_generated_tools.validate_files([path], issue_text=self.UBER_STAGE_ISSUE)
        self.assertEqual(failures, [])

    def test_static_contract_rejects_video_config(self):
        issue = "## Mode\n\ntake_photo\n"
        tool = '''
from litellm_utils import call_take_photo_vlm
TOOL_NAME = "hand_identifier"
EXECUTION_MODE = "take_photo"
VIDEO_CONFIG = {"window_seconds": 6}
TOOL_PROMPT = "Identify the current hand."
def main(image, input_data):
    return call_take_photo_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
'''
        failures = validate_generated_tools.validate_take_photo_tool(
            tool, issue, Path("tools/hand_identifier.py")
        )
        self.assertTrue(any("must not declare" in failure for failure in failures))

    def test_temporal_contract_requires_complete_window_and_temporal_prompt(self):
        issue = "## Mode\n\nhosted_video_streaming\n"
        incomplete = '''
TOOL_NAME = "door_change"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {"window_seconds": 6}
TOOL_PROMPT = "Describe the door."
'''
        failures = validate_generated_tools.validate_rtvi_streaming_tool(
            incomplete, issue, Path("tools/door_change.py")
        )
        self.assertTrue(any("VIDEO_CONFIG is missing" in failure for failure in failures))
        self.assertTrue(any("chronological or state-change" in failure for failure in failures))

        complete = '''
TOOL_NAME = "door_change"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "minimum_span_seconds": 5,
    "minimum_unique_frames": 12,
}
TOOL_PROMPT = "Compare chronological early and late frames and report whether the door changed."
'''
        self.assertEqual(
            validate_generated_tools.validate_rtvi_streaming_tool(
                complete, issue, Path("tools/door_change.py")
            ),
            [],
        )

    def test_temporal_sign_language_tool_validates_as_ordered_video(self):
        issue = """## Task
Understand a temporal short sign-language gesture from recent hand movements.

## Expected output
Identify the sign or short signed phrase they just made, including signs lasting seconds.

## Mode

hosted_video_streaming
"""
        tool = '''
TOOL_NAME = "recent_sign_language"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "minimum_span_seconds": 4,
    "minimum_unique_frames": 8,
}
OUTPUT_CONFIG = {"deduplicate": True}
TOOL_PROMPT = (
    "Inspect the four chronological frames from early to late. Track the signer's "
    "hand shapes, orientation, location, and motion sequence over the recent interval, "
    "then identify the completed sign or short signed phrase."
)
'''
        self.assertEqual(
            validate_generated_tools.validate_rtvi_streaming_tool(
                tool, issue, Path("tools/recent_sign_language.py")
            ),
            [],
        )

    def test_validation_follows_copilot_declaration_not_parser_mode_suggestion(self):
        issue_with_wrong_suggestion = """## Task
Identify the signed phrase from recent movement.

## Mode

take_photo
"""
        temporal_tool = '''
TOOL_NAME = "recent_sign_language"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "minimum_span_seconds": 4,
    "minimum_unique_frames": 8,
}
TOOL_PROMPT = "Compare chronological early and late hand movement and identify the signed phrase."
'''
        path = Path("tools/recent_sign_language.py")
        self.assertEqual(
            validate_generated_tools.validate_take_photo_tool(
                temporal_tool, issue_with_wrong_suggestion, path
            ),
            [],
        )
        self.assertEqual(
            validate_generated_tools.validate_rtvi_streaming_tool(
                temporal_tool, issue_with_wrong_suggestion, path
            ),
            [],
        )

    def test_sign_language_tool_cannot_reuse_played_card_schema(self):
        issue = "## Mode\n\nhosted_video_streaming\n"
        tool = '''
TOOL_NAME = "recent_sign_language"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "minimum_span_seconds": 4,
    "minimum_unique_frames": 8,
}
OUTPUT_CONFIG = {"schema": "played_card_event"}
TOOL_PROMPT = "Inspect chronological hand motion and identify the signed phrase."
'''
        failures = validate_generated_tools.validate_rtvi_streaming_tool(
            tool, issue, Path("tools/recent_sign_language.py")
        )
        self.assertTrue(any("card-specific prompt" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
