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
        raw = model_router._load_yaml(model_router.CAPABILITY_PROFILES_PATH)["capabilities"]

        self.assertIn("object_detection", capabilities)
        self.assertIn("ocr", capabilities)
        self.assertIsInstance(raw["ocr"], dict)
        self.assertIn("description", raw["ocr"])
        self.assertIn("include_examples", raw["ocr"])
        self.assertIn("exclude_examples", raw["ocr"])
        self.assertTrue(len(capabilities["map_web"]) >= 3)
        self.assertIn("YOLO-World", {profile.name for profile in profiles})
        self.assertIn("GoogleVisionOCR", {profile.name for profile in profiles})

    def test_benchmark_profiles_use_benchmark_max_scaling(self):
        profiles = {profile.name: profile for profile in model_router.load_model_profiles()}

        self.assertEqual(profiles["LLaVA-OneVision-7B"].capabilities["general_reasoning"], 0.619)
        self.assertEqual(profiles["Qwen2.5-VL-7B"].capabilities["general_reasoning"], 0.641)
        self.assertEqual(profiles["Gemini-2.5-pro"].capabilities["general_reasoning"], 0.736)

    def test_compute_capability_weights_uses_general_reasoning_fallback(self):
        weights = model_router.compute_capability_weights("Locate a specific item in the scene.")

        self.assertEqual(weights, {"general_reasoning": 1.0})

    def test_compute_capability_weights_uses_first_capability_if_general_missing(self):
        descriptions = {
            "ocr": ["Read text from images"],
            "object_detection": ["Detect objects"],
        }
        weights = model_router.compute_capability_weights("Read label", descriptions)

        self.assertEqual(weights, {"ocr": 1.0})

    def test_rank_models_prefers_routing_analysis_over_fallback(self):
        profiles = [
            model_router.ModelProfile("ocr_model", "general_vlm", 1000, "test", {"ocr": 0.95, "general_reasoning": 0.1}, latency=0.6),
            model_router.ModelProfile("general_model", "general_vlm", 1000, "test", {"ocr": 0.1, "general_reasoning": 0.95}, latency=0.6),
        ]
        routing_analysis = {
            "tasks": [{"name": "ocr", "weight": 1.0, "reason": "Read text"}],
            "latency_sensitivity": {"level": "medium", "weight": 0.5, "reason": ""},
        }

        result = model_router.rank_models("Read this label.", profiles, routing_analysis=routing_analysis)

        self.assertEqual(result["selected_model"], "ocr_model")

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

    def test_rank_models_prefers_fast_model_when_latency_is_high(self):
        profiles = [
            model_router.ModelProfile(
                "accurate_but_slow",
                "general_vlm",
                2500,
                "test",
                {"ocr": 0.95, "general_reasoning": 0.95},
                latency=0.3,
            ),
            model_router.ModelProfile(
                "balanced_fast",
                "general_vlm",
                600,
                "test",
                {"ocr": 0.85, "general_reasoning": 0.8},
                latency=0.9,
            ),
        ]
        routing_analysis = {
            "tasks": [
                {"name": "ocr", "weight": 0.6, "reason": "Read text."},
                {"name": "general_reasoning", "weight": 0.4, "reason": "Explain clearly."},
            ],
            "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": "Live feedback."},
        }

        result = model_router.rank_models("Read this label quickly.", profiles, routing_analysis=routing_analysis)

        self.assertEqual(result["selected_model"], "balanced_fast")
        self.assertIsNotNone(result.get("routing_analysis"))

    def test_rank_models_falls_back_when_routing_analysis_invalid(self):
        descriptions = {"ocr": ["Read text from an image."]}
        profiles = [
            model_router.ModelProfile("strong_ocr", "general_vlm", 1000, "test", {"ocr": 1.0}),
            model_router.ModelProfile("weak_fast_ocr", "general_vlm", 10, "test", {"ocr": 0.7}),
        ]

        result = model_router.rank_models(
            "Read text from an image.",
            profiles,
            descriptions,
            routing_analysis={"tasks": [{"name": "unknown_task", "weight": 1.0}]},
        )

        self.assertEqual(result["selected_model"], "strong_ocr")
        self.assertIsNone(result.get("routing_analysis"))

    def test_default_routing_without_analysis_uses_general_fallback(self):
        result = model_router.select_model("Any task without parse agent routing analysis")

        self.assertEqual(result["capability_weights"], {"general_reasoning": 1.0})
        self.assertIsNotNone(result["selected_model"])
        self.assertGreaterEqual(len(result["ranking"]), 1)

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
