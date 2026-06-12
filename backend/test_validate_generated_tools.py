"""Tests for generated-tool router guardrails."""

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_generated_tools


class TestValidateGeneratedTools(unittest.TestCase):
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
from model_router_client import llm_call

TASK_CATEGORY = "visual_reasoning"

def main(image, input_data=None):
    return llm_call(
        task_category=TASK_CATEGORY,
        messages=[{"role": "user", "content": "Describe the relevant scene details."}],
        images=[image],
        metadata={"tool_name": "good_generated_tool"},
    )
""",
        )

        self.assertEqual(validate_generated_tools.validate_files([path]), [])


if __name__ == "__main__":
    unittest.main()
