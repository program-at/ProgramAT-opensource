"""Tests for the semantic capability router."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_router


class TestSemanticCapabilityRouter(unittest.TestCase):
    def test_loads_profiles_and_capabilities_from_yaml(self):
        capabilities = model_router.load_capability_descriptions()
        profiles = model_router.load_model_profiles()

        self.assertIn("object_detection", capabilities)
        self.assertIn("ocr", capabilities)
        self.assertEqual(
            capabilities["map_web"],
            [
                "Interpret a map or route diagram.",
                "Read charts, tables, timetables, and schedules.",
                "Understand webpage and app layouts.",
                "Describe structured visual layouts such as calendars and departure boards.",
            ],
        )
        self.assertIn("YOLO-World", {profile.name for profile in profiles})
        self.assertIn("GoogleVisionOCR", {profile.name for profile in profiles})

    def test_benchmark_profiles_use_benchmark_max_scaling(self):
        profiles = {profile.name: profile for profile in model_router.load_model_profiles()}

        self.assertEqual(profiles["LLaVA-OneVision-7B"].capabilities["ocr"], 0.622)
        self.assertEqual(profiles["Qwen2.5-VL-7B"].capabilities["ocr"], 0.888)
        self.assertEqual(profiles["Gemini-2.5-pro"].capabilities["general_reasoning"], 0.736)

    def test_compute_capability_weights_uses_task_text(self):
        weights = model_router.compute_capability_weights("Locate a specific item in the scene.")

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(max(weights, key=weights.get), "object_detection")
        self.assertGreater(weights["object_detection"], weights["ocr"])

    def test_compute_capability_weights_uses_top_two_average(self):
        descriptions = {
            "ocr": ["ocr one", "ocr two", "ocr three"],
            "object_detection": ["object one"],
        }

        with patch.object(model_router, "_capability_similarities", return_value={
            "ocr": [0.95, 0.90, 0.10],
            "object_detection": [0.80],
        }):
            weights = model_router.compute_capability_weights("Read the label.", descriptions)

        self.assertGreater(weights["ocr"], weights["object_detection"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_compute_capability_weights_applies_keyword_boosts(self):
        descriptions = {
            "ocr": ["ocr"],
            "object_detection": ["object"],
            "navigation": ["navigation"],
            "spatial_relationship": ["spatial"],
            "general_reasoning": ["general"],
        }

        with patch.object(model_router, "_capability_similarities", return_value={
            "ocr": [0.20],
            "object_detection": [0.20],
            "navigation": [0.20],
            "spatial_relationship": [0.20],
            "general_reasoning": [0.20],
        }):
            weights = model_router.compute_capability_weights(
                "Read the sign and go to the exit on the left.",
                descriptions,
            )

        self.assertEqual(max(weights, key=weights.get), "ocr")
        self.assertGreater(weights["navigation"], weights["general_reasoning"])
        self.assertGreater(weights["spatial_relationship"], weights["general_reasoning"])

    def test_score_model_treats_missing_capabilities_as_zero(self):
        profile = model_router.ModelProfile(
            name="narrow",
            type="specialized_expert",
            latency_ms=100,
            source="test",
            capabilities={"object_detection": 1.0},
        )

        score = model_router.score_model(profile, {"object_detection": 0.4, "ocr": 0.6})

        self.assertEqual(score, 0.4)

    def test_rank_models_prefers_quality_before_latency(self):
        descriptions = {"ocr": ["Read text from an image."]}
        profiles = [
            model_router.ModelProfile("strong_ocr", "general_vlm", 1000, "test", {"ocr": 1.0}),
            model_router.ModelProfile("weak_fast_ocr", "general_vlm", 10, "test", {"ocr": 0.7}),
        ]

        result = model_router.rank_models("Read text from an image.", profiles, descriptions)

        self.assertEqual(result["selected_model"], "strong_ocr")
        self.assertGreater(result["selected"]["capability_score"], 0)

    def test_rank_models_uses_latency_when_capability_is_equal(self):
        descriptions = {"ocr": ["Read text from an image."]}
        profiles = [
            model_router.ModelProfile("slow_ocr", "general_vlm", 2000, "test", {"ocr": 0.9}),
            model_router.ModelProfile("fast_ocr", "general_vlm", 100, "test", {"ocr": 0.9}),
        ]

        result = model_router.rank_models("Read text from an image.", profiles, descriptions)

        self.assertEqual(result["selected_model"], "fast_ocr")

    def test_default_routing_examples(self):
        examples = {
            "Read the medicine bottle label.": "Qwen2.5-VL-7B",
            "What does this sign say?": "GoogleVisionOCR",
            "Help me find my cup.": "YOLO-World",
            "Locate the passenger door handle.": "YOLO-World",
            "Describe what is happening in front of me.": "Gemini-2.0-flash",
            "Guide me to the building entrance.": "Gemini-2.5-pro",
            "Explain the relationship between the chair and the table.": "Gemini-2.0-flash",
            "Analyze this video.": "Qwen2.5-VL-7B",
        }

        for task, selected_model in examples.items():
            with self.subTest(task=task):
                result = model_router.select_model(task)
                self.assertEqual(result["selected_model"], selected_model)
                self.assertAlmostEqual(sum(result["capability_weights"].values()), 1.0)
                self.assertGreaterEqual(len(result["ranking"]), 3)
                self.assertEqual(result["ranking"][0]["model"], selected_model)

    def test_system_llm_call_uses_fixed_model_without_router(self):
        with patch.object(model_router, "call_model", return_value={"choices": []}) as call_model, \
             patch.object(model_router, "rank_models") as rank_models:
            model_router.system_llm_call(messages=[{"role": "user", "content": "Parse this."}])

        rank_models.assert_not_called()
        self.assertEqual(call_model.call_args.args[0], model_router.SYSTEM_MODEL)

    def test_copilot_llm_call_routes_then_calls_selected_profile_model(self):
        profiles = [
            model_router.ModelProfile(
                name="FastCopilot",
                type="general_vlm",
                latency_ms=100,
                source="test",
                capabilities={"general_reasoning": 1.0},
                model="gemini/gemini-2.0-flash",
            )
        ]
        route = {
            "selected_model": "FastCopilot",
            "capability_weights": {"general_reasoning": 1.0},
        }

        with patch.object(model_router, "load_model_profiles", return_value=profiles), \
             patch.object(model_router, "rank_models", return_value=route) as rank_models, \
             patch.object(model_router, "call_model", return_value={"choices": []}) as call_model:
            model_router.copilot_llm_call(
                task="code_generation",
                messages=[{"role": "user", "content": "Build a tool."}],
            )

        rank_models.assert_called_once()
        self.assertEqual(call_model.call_args.args[0], "gemini/gemini-2.0-flash")


if __name__ == "__main__":
    unittest.main()
