"""Tests for semantic model routing."""

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

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

        self.assertEqual(route["selected_profile"], "llama")
        self.assertEqual(route["selected_model"], "groq/llama-3.1-8b-instant")

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

    def test_routes_new_task_categories(self):
        simple_route = model_router.get_route_info(
            "simple_parsing",
            {"route_text": "classify a short command and extract fields"},
        )
        ocr_route = model_router.get_route_info(
            "OCR",
            {"route_text": "read text from a sign in the camera frame"},
        )
        detection_route = model_router.get_route_info(
            "object_detection",
            {"labels": ["car"], "route_text": "detect cars with bounding boxes"},
        )
        world_route = model_router.get_route_info(
            "object_localization",
            {"labels": ["door handle"], "route_text": "localize a door handle"},
        )
        visual_reasoning_route = model_router.get_route_info(
            "visual_reasoning",
            {"route_text": "reason about the spatial layout of objects in an image"},
        )

        self.assertEqual(simple_route["task_category"], "simple_parsing")
        self.assertEqual(simple_route["selected_profile"], "gemini_flash_lite")
        self.assertEqual(ocr_route["task_category"], "ocr")
        self.assertEqual(ocr_route["selected_profile"], "google_vision_ocr")
        self.assertEqual(detection_route["selected_profile"], "yolo11_detector")
        self.assertEqual(world_route["selected_profile"], "yolo_world_detector")
        self.assertEqual(visual_reasoning_route["task_category"], "visual_reasoning")
        self.assertEqual(visual_reasoning_route["selected_profile"], "gpt4o")

    def test_llm_call_accepts_task_alias(self):
        fake_litellm = SimpleNamespace()

        with patch.object(model_router, "LITELLM_AVAILABLE", True), \
             patch.object(model_router, "litellm", fake_litellm), \
             patch.object(fake_litellm, "completion", return_value={"choices": []}, create=True) as completion:
            model_router.llm_call(
                task="visual_understanding",
                messages=[{"role": "user", "content": "Describe the clothing."}],
                metadata={
                    "tool_name": "clothing_recognition",
                    "route_text": "identify clothing in a camera frame",
                },
            )

        self.assertEqual(completion.call_args.kwargs["model"], "openai/gpt-4o")

    def test_llm_call_encodes_numpy_ndarray_image(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        fake_litellm = SimpleNamespace()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]

        with patch.object(model_router, "LITELLM_AVAILABLE", True), \
             patch.object(model_router, "litellm", fake_litellm), \
             patch.object(fake_litellm, "completion", return_value={"choices": []}, create=True) as completion:
            model_router.llm_call(
                task_category="visual_understanding",
                messages=[{"role": "user", "content": "Describe this image."}],
                images=[image],
                metadata={
                    "tool_name": "ndarray_image_test",
                    "route_text": "describe a camera frame",
                },
            )

        content = completion.call_args.kwargs["messages"][-1]["content"]
        image_part = next(part for part in content if part.get("type") == "image_url")
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_llm_call_falls_back_from_non_llm_backend(self):
        fake_litellm = SimpleNamespace()

        with patch.object(model_router, "LITELLM_AVAILABLE", True), \
             patch.object(model_router, "litellm", fake_litellm), \
             patch.object(fake_litellm, "completion", return_value={"choices": []}, create=True) as completion:
            model_router.llm_call(
                task="ocr",
                messages=[{"role": "user", "content": "Clean up OCR text."}],
                metadata={"tool_name": "ocr_cleanup"},
            )

        self.assertEqual(completion.call_args.kwargs["model"], "gemini/gemini-2.5-flash-lite")

    def test_vision_call_routes_images_to_llm_call(self):
        with patch.object(model_router, "llm_call", return_value={"choices": []}) as llm_call:
            model_router.vision_call(
                task="visual_reasoning",
                image="abc123",
                prompt="Find the correct vehicle.",
                metadata={"tool_name": "uber_finder"},
            )

        self.assertEqual(llm_call.call_args.kwargs["task"], "visual_reasoning")
        self.assertEqual(llm_call.call_args.kwargs["images"], ["abc123"])

    def test_detect_objects_routes_to_detector_backend(self):
        with patch.object(model_router, "_run_ultralytics_detector", return_value=[]) as run_detector:
            result = model_router.detect_objects(
                task="object_detection",
                image=object(),
                labels=["car"],
                metadata={"tool_name": "car_counter"},
            )

        self.assertEqual(result, [])
        self.assertEqual(run_detector.call_args.kwargs["model_name"], "local/yolo11n.pt")

    def test_unknown_category_reports_fallback_model(self):
        original_min_score = model_router.ROUTING_MIN_SCORE
        model_router.ROUTING_MIN_SCORE = 1.0
        try:
            route = model_router.get_route_info(
                "unknown_category",
                {"route_text": "something unrelated"},
            )
        finally:
            model_router.ROUTING_MIN_SCORE = original_min_score

        self.assertEqual(route["fallback_profile"], "gemini_flash")
        self.assertEqual(route["fallback_model"], "gemini/gemini-3-flash-preview")
        self.assertIsNotNone(route["fallback_reason"])

    def test_explicit_model_override_still_wins(self):
        route = model_router.get_route_info(
            "image_analysis",
            {
                "requested_model": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
                "route_text": "answer a visual question about an image",
            },
        )

        self.assertEqual(route["selected_profile"], "llava")
        self.assertEqual(route["selected_model"], "groq/meta-llama/llama-4-scout-17b-16e-instruct")


if __name__ == "__main__":
    unittest.main()
