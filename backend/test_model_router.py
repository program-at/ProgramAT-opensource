"""Tests for semantic model routing."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_router


class TestSemanticModelRouter(unittest.TestCase):
    def setUp(self):
        self.original_mode = model_router.ROUTING_MODE
        model_router.ROUTING_MODE = "semantic"

    def tearDown(self):
        model_router.ROUTING_MODE = self.original_mode

    def test_routes_visual_requests_directly_to_vision_model(self):
        route = model_router.get_route_info(
            "image_analysis",
            {"route_text": "answer a user's follow-up question about an image from the camera frame"},
        )

        self.assertEqual(route["selected_profile"], "gpt4o")

    def test_routes_code_requests_directly_to_coding_model(self):
        route = model_router.get_route_info(
            "code_generation",
            {"route_text": "generate Python code for a new assistive technology tool and fix failing tests"},
        )

        self.assertEqual(route["selected_profile"], "claude")

    def test_routes_text_json_requests_directly_to_fast_lite_model(self):
        route = model_router.get_route_info(
            "text_parse",
            {"route_text": "parse a voice transcript into structured JSON fields for an issue"},
        )

        self.assertEqual(route["selected_profile"], "gemini_flash_lite")
        self.assertEqual(route["selected_model"], "gemini/gemini-2.5-flash-lite")

    def test_routes_summaries_directly_to_fast_lite_model(self):
        route = model_router.get_route_info(
            "summarization",
            {"route_text": "summarize Copilot logs concisely for text to speech"},
        )

        self.assertEqual(route["selected_profile"], "gemini_flash_lite")

    def test_explicit_model_override_still_wins(self):
        route = model_router.get_route_info(
            "image_analysis",
            {
                "requested_model": "anthropic/claude-3-5-sonnet-20241022",
                "route_text": "answer a visual question about an image",
            },
        )

        self.assertEqual(route["selected_profile"], "claude")


if __name__ == "__main__":
    unittest.main()
