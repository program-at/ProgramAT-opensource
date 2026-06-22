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
Capability: object_detection

Stage 2:
Capability: spatial_relationship

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

    def test_allows_model_router_client_import(self):
        path = self._write_temp_tool(
            "good_generated_tool.py",
            """
from model_router_client import copilot_llm_call

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

    def test_rejects_non_canonical_task_category_visual_reasoning(self):
        path = self._write_temp_tool(
            "bad_visual_reasoning_tool.py",
            """
from model_router_client import copilot_llm_call

def main(image, input_data=None):
    return copilot_llm_call(
        task_category="visual_reasoning",
        messages=[{"role": "user", "content": "Find my Uber and guide me."}],
        images=[image],
        metadata={"tool_name": "bad_visual_reasoning_tool"},
    )
""",
        )

        failures = validate_generated_tools.validate_files([path])
        self.assertTrue(any("non-canonical task_category 'visual_reasoning'" in failure for failure in failures))

    def test_rejects_single_call_when_issue_has_three_stages(self):
        path = self._write_temp_tool(
            "uber_single_call_tool.py",
            """
from model_router_client import copilot_llm_call

def main(image, input_data=None):
    return copilot_llm_call(
        task_category="general_reasoning",
        messages=[{"role": "user", "content": "Handle all Uber steps in one call."}],
        images=[image],
        metadata={"tool_name": "uber_single_call_tool"},
    )
""",
        )

        failures = validate_generated_tools.validate_files([path], issue_text=self.UBER_STAGE_ISSUE)

        self.assertTrue(any("Task Stages lists 3 model-backed stage(s), but tool has only 1" in failure for failure in failures))
        self.assertTrue(any("Task Stage 1 requires task_category 'object_detection'" in failure for failure in failures))

    def test_allows_three_calls_matching_uber_stage_capabilities(self):
        path = self._write_temp_tool(
            "uber_three_stage_tool.py",
            """
from model_router_client import copilot_llm_call

def main(image, input_data=None):
    vehicle_result = copilot_llm_call(
        task_category="object_detection",
        messages=[{"role": "user", "content": "Locate the right vehicle."}],
        images=[image],
        metadata={"tool_name": "uber_three_stage_tool", "stage_name": "Stage 1"},
    )
    door_result = copilot_llm_call(
        task_category="spatial_relationship",
        messages=[{"role": "user", "content": "Locate the passenger-side door."}],
        images=[image],
        metadata={
            "tool_name": "uber_three_stage_tool",
            "stage_name": "Stage 2",
            "previous_stage_output": vehicle_result,
        },
    )
    guidance_result = copilot_llm_call(
        task_category="navigation",
        messages=[{"role": "user", "content": "Guide user to the door."}],
        images=[image],
        metadata={
            "tool_name": "uber_three_stage_tool",
            "stage_name": "Stage 3",
            "previous_stage_outputs": {"vehicle": vehicle_result, "door": door_result},
        },
    )
    return guidance_result
""",
        )

        failures = validate_generated_tools.validate_files([path], issue_text=self.UBER_STAGE_ISSUE)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
