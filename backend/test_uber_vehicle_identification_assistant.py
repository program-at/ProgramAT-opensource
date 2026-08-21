"""Tests for uber_vehicle_identification_assistant tool."""

from __future__ import annotations

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
fake_router_client = ModuleType("model_router_client")
fake_router_client.copilot_llm_call = lambda **_: {"response": ""}
sys.modules.setdefault("model_router_client", fake_router_client)

import uber_vehicle_identification_assistant as tool


def _test_image() -> np.ndarray:
    return np.ones((120, 160, 3), dtype=np.uint8) * 127


class TestUberVehicleIdentificationAssistant(unittest.TestCase):
    def test_no_image_returns_audio_error(self):
        result = tool.main(None, {})
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("audio", {}).get("type"), "error")

    def test_uses_ordered_stages_and_passes_usable_artifacts(self):
        mocked_results = [
            {"response": "vehicle located", "artifact": {"detections": [{"label": "car"}], "confidence": 0.91}},
            {"response": "ABC123", "artifact": {"text": "ABC123", "confidence": 0.84}},
            {"response": "White Toyota Camry, plate ABC123. High match confidence."},
        ]
        with patch.object(tool, "copilot_llm_call", side_effect=mocked_results) as mock_call:
            result = tool.main(_test_image(), {"expected_make": "Toyota", "expected_color": "white"})

        self.assertIn("Toyota Camry", result)
        self.assertEqual(mock_call.call_count, 3)
        self.assertEqual(mock_call.call_args_list[0].kwargs["capability"], "object_detection_localization")
        self.assertEqual(mock_call.call_args_list[1].kwargs["capability"], "ocr")
        self.assertEqual(mock_call.call_args_list[2].kwargs["capability"], "general_reasoning")
        self.assertIn("previous_stage_artifact", mock_call.call_args_list[1].kwargs["metadata"])
        self.assertIn("previous_stage_artifact", mock_call.call_args_list[2].kwargs["metadata"])

    def test_skips_failed_artifacts_for_later_stages(self):
        mocked_results = [
            {"response": "uncertain", "artifact": {}},
            {"response": "uncertain ocr", "artifact": {"text": "", "confidence": 0.95}},
            {"response": "I cannot confirm the vehicle yet."},
        ]
        with patch.object(tool, "copilot_llm_call", side_effect=mocked_results) as mock_call:
            result = tool.main(_test_image(), {"query": "black Honda"})

        self.assertEqual(result, "I cannot confirm the vehicle yet.")
        self.assertNotIn("previous_stage_artifact", mock_call.call_args_list[1].kwargs["metadata"])
        self.assertNotIn("previous_stage_artifact", mock_call.call_args_list[2].kwargs["metadata"])

    def test_streaming_response_is_limited_to_15_words(self):
        long_response = (
            "White Toyota Camry center frame plate ABC123 likely your Uber based on color and make details."
        )
        mocked_results = [
            {"response": "vehicle located", "artifact": {"detections": [{"label": "car"}], "confidence": 0.92}},
            {"response": "ABC123", "artifact": {"text": "ABC123", "confidence": 0.88}},
            {"response": long_response},
        ]
        with patch.object(tool, "copilot_llm_call", side_effect=mocked_results):
            result = tool.main(_test_image(), {"streaming": True})

        self.assertLessEqual(len(result.split()), 15)

    def test_reasoning_receives_partial_artifacts_when_available(self):
        mocked_results = [
            {"response": "vehicle located", "artifact": {"detections": [{"label": "car"}], "confidence": 0.94}},
            {"response": "no text", "artifact": {"text": "", "confidence": 0.97}},
            {"response": "Vehicle appears to be a white Toyota."},
        ]
        with patch.object(tool, "copilot_llm_call", side_effect=mocked_results) as mock_call:
            tool.main(_test_image(), {"expected_make": "Toyota"})

        handed_off = mock_call.call_args_list[2].kwargs["metadata"].get("previous_stage_artifact", {})
        self.assertIn("vehicle_detection", handed_off)
        self.assertNotIn("ocr", handed_off)

    def test_value_field_is_used_as_query_fallback(self):
        mocked_results = [
            {"response": "vehicle located", "artifact": {"detections": [{"label": "car"}], "confidence": 0.9}},
            {"response": "ABC123", "artifact": {"text": "ABC123", "confidence": 0.8}},
            {"response": "Black Honda Civic, plate ABC123. Possibly your ride."},
        ]
        with patch.object(tool, "copilot_llm_call", side_effect=mocked_results) as mock_call:
            tool.main(_test_image(), {"value": "black honda"})

        self.assertEqual(mock_call.call_count, 3)
        reasoning_messages = mock_call.call_args_list[2].kwargs.get("messages")
        self.assertIsNotNone(reasoning_messages)
        self.assertIn("User request: black honda", reasoning_messages[1]["content"])

    def test_runtime_input_block_key_is_honored(self):
        mocked_results = [
            {"response": "vehicle located", "artifact": {"detections": [{"label": "car"}], "confidence": 0.9}},
            {"response": "ABC123", "artifact": {"text": "ABC123", "confidence": 0.8}},
            {"response": "Black Honda Civic, plate ABC123. Possibly your ride."},
        ]
        with patch.object(tool, "copilot_llm_call", side_effect=mocked_results) as mock_call, patch.object(
            tool, "TOOL_RUNTIME_INPUT", {"key": "uber_query"}
        ), patch.object(tool, "TOOL_RUNTIME_INPUT_KEY", "uber_query"):
            tool.main(_test_image(), {"uber_query": "blue toyota"})
            reasoning_messages = mock_call.call_args_list[2].kwargs.get("messages")
            self.assertIsNotNone(reasoning_messages)
            self.assertIn("User request: blue toyota", reasoning_messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
