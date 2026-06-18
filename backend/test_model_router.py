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

    def test_pipeline_analysis_false_keeps_single_model_plan(self):
        profiles = [
            model_router.ModelProfile(
                "general_model",
                "general_vlm",
                1000,
                "test",
                {"general_reasoning": 0.9, "object_detection": 0.8},
                latency=0.7,
            ),
            model_router.ModelProfile(
                "detector",
                "specialized_expert",
                80,
                "test",
                {"object_detection": 1.0, "general_reasoning": 0.0},
                latency=0.98,
            ),
        ]
        routing_analysis = {
            "tasks": [
                {"name": "general_reasoning", "weight": 0.7, "reason": "Holistic scene summary."},
                {"name": "object_detection", "weight": 0.3, "reason": "Notice key objects."},
            ],
            "latency_sensitivity": {"level": "medium", "weight": 0.5, "reason": ""},
        }
        pipeline_analysis = {
            "should_chain": False,
            "reason": "The later model would still need the full image.",
            "stages": [],
        }

        result = model_router.rank_models(
            "Describe this room and mention important objects.",
            profiles,
            routing_analysis=routing_analysis,
            pipeline_analysis=pipeline_analysis,
        )

        self.assertEqual(result["selected_model"], "general_model")
        self.assertFalse(result["pipeline_analysis"]["should_chain"])
        self.assertEqual(result["stage_model_selection"], [])
        self.assertEqual(result["final_execution_plan"]["mode"], "single_model")

    def test_pipeline_analysis_true_selects_model_per_stage(self):
        profiles = [
            model_router.ModelProfile(
                "general_navigation",
                "general_vlm",
                1200,
                "test",
                {"navigation": 0.9, "object_detection": 0.2},
                latency=0.7,
            ),
            model_router.ModelProfile(
                "fast_detector",
                "specialized_expert",
                80,
                "test",
                {"object_detection": 1.0, "navigation": 0.0},
                latency=0.98,
            ),
            model_router.ModelProfile(
                "weak_allrounder",
                "general_vlm",
                300,
                "test",
                {"object_detection": 0.4, "navigation": 0.4},
                latency=0.9,
            ),
        ]
        routing_analysis = {
            "tasks": [
                {"name": "object_detection", "weight": 0.5, "reason": "Find the door."},
                {"name": "navigation", "weight": 0.5, "reason": "Guide the user to it."},
            ],
            "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": "Live guidance."},
        }
        pipeline_analysis = {
            "should_chain": True,
            "reason": "A door bounding box reduces navigation work.",
            "stages": [
                {
                    "capability": "object_detection",
                    "purpose": "Find the door.",
                    "input": "Full image.",
                    "output": "Door bounding box.",
                    "preferred_model_type": "fast_detector",
                },
                {
                    "capability": "navigation",
                    "purpose": "Guide the user to the detected door.",
                    "input": "Door bounding box and current frame.",
                    "output": "Movement guidance.",
                    "preferred_model_type": "reasoning_vlm",
                },
            ],
        }

        result = model_router.rank_models(
            "Find the door and guide me there.",
            profiles,
            routing_analysis=routing_analysis,
            pipeline_analysis=pipeline_analysis,
        )

        self.assertIsNone(result["selected_model"])
        self.assertTrue(result["pipeline_analysis"]["should_chain"])
        self.assertEqual(result["final_execution_plan"]["mode"], "pipeline")
        self.assertEqual(
            [stage["selected_model"] for stage in result["stage_model_selection"]],
            ["fast_detector", "general_navigation"],
        )

    def test_pipeline_analysis_requires_two_valid_stages(self):
        result = model_router.normalize_pipeline_analysis(
            {
                "should_chain": True,
                "reason": "Only one valid stage was provided.",
                "stages": [
                    {"capability": "object_detection", "purpose": "Find objects."},
                    {"capability": "unknown", "purpose": "Invalid."},
                ],
            },
            supported_capabilities=["object_detection", "navigation"],
        )

        self.assertFalse(result["should_chain"])
        self.assertEqual(result["stages"], [])

    def test_information_reduction_infers_find_then_guide_pipeline(self):
        profiles = [
            model_router.ModelProfile(
                "navigator",
                "general_vlm",
                1000,
                "test",
                {"navigation": 0.9, "object_detection": 0.1},
                latency=0.7,
            ),
            model_router.ModelProfile(
                "detector",
                "specialized_expert",
                80,
                "test",
                {"object_detection": 1.0, "navigation": 0.0},
                latency=0.98,
            ),
        ]
        routing_analysis = {
            "tasks": [{"name": "navigation", "weight": 1.0, "reason": "Guide to elevator."}],
            "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": "Live navigation."},
        }
        pipeline_analysis = {
            "should_chain": False,
            "reason": "A single VLM can do it.",
            "stages": [],
        }

        result = model_router.rank_models(
            "Find the elevator and guide me to it.",
            profiles,
            {"object_detection": ["Detect objects"], "navigation": ["Navigate"]},
            routing_analysis=routing_analysis,
            pipeline_analysis=pipeline_analysis,
        )

        self.assertTrue(result["pipeline_analysis"]["should_chain"])
        self.assertEqual(result["final_execution_plan"]["mode"], "pipeline")
        self.assertEqual(
            [stage["capability"] for stage in result["pipeline_analysis"]["stages"]],
            ["object_detection", "navigation"],
        )

    def test_information_reduction_infers_bus_entrance_pipeline(self):
        result = model_router.infer_pipeline_from_information_reduction(
            "Identify the correct bus and tell me where the entrance is.",
            {
                "tasks": [
                    {"name": "map_web", "weight": 0.6, "reason": "Identify correct bus."},
                    {"name": "object_detection", "weight": 0.4, "reason": "Find entrance."},
                ],
                "latency_sensitivity": {"level": "high", "weight": 0.8, "reason": ""},
            },
            {"should_chain": False, "reason": "Too conservative.", "stages": []},
            ["object_detection", "map_web", "spatial_relationship"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(result["stages"][0]["capability"], "object_detection")
        self.assertEqual(result["stages"][1]["capability"], "map_web")

    def test_information_reduction_infers_parent_object_localization_pipeline(self):
        result = model_router.infer_pipeline_from_information_reduction(
            "Find the passenger-side door handle.",
            {
                "tasks": [
                    {"name": "object_detection", "weight": 0.7, "reason": "Find handle."},
                    {"name": "general_reasoning", "weight": 0.3, "reason": "Reason about passenger side."},
                ],
                "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": ""},
            },
            {"should_chain": False, "reason": "Too conservative.", "stages": []},
            ["object_detection", "spatial_relationship", "general_reasoning"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(
            [stage["capability"] for stage in result["stages"]],
            ["object_detection", "general_reasoning"],
        )

    def test_information_flow_camera_guidance_uses_detection_then_camera_motion(self):
        result = model_router.infer_pipeline_from_information_reduction(
            "Guide my camera toward the target.",
            {
                "tasks": [
                    {"name": "navigation", "weight": 0.7, "reason": "guidance toward target"},
                    {"name": "camera_motion", "weight": 0.3, "reason": "camera movement guidance"},
                ],
                "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": "live guidance"},
            },
            {"should_chain": False, "reason": "Too conservative.", "stages": []},
            ["object_detection", "navigation", "camera_motion", "spatial_relationship"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(
            [stage["capability"] for stage in result["stages"]],
            ["object_detection", "camera_motion"],
        )

    def test_information_flow_find_then_guide_prefers_navigation_not_camera_motion(self):
        result = model_router.infer_pipeline_from_information_reduction(
            "Find the elevator and guide me to it.",
            {
                "tasks": [
                    {"name": "navigation", "weight": 0.7, "reason": "wayfinding guidance"},
                    {"name": "camera_motion", "weight": 0.3, "reason": "camera alignment"},
                ],
                "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": "live guidance"},
            },
            {"should_chain": False, "reason": "Too conservative.", "stages": []},
            ["object_detection", "navigation", "camera_motion"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(
            [stage["capability"] for stage in result["stages"]],
            ["object_detection", "navigation"],
        )

    def test_pipeline_normalization_uses_complexity_penalty_without_stage_cap(self):
        result = model_router.normalize_pipeline_analysis(
            {
                "should_chain": True,
                "reason": "Multiple useful waypoints preserve artifact flow.",
                "artifact_value_score": 1.0,
                "information_reduction": "significant",
                "stages": [
                    {"capability": "object_detection", "purpose": "Find the office.", "input": "scene", "output": "office bbox"},
                    {"capability": "spatial_relationship", "purpose": "Locate stairs near office.", "input": "office bbox", "output": "stairs waypoint"},
                    {"capability": "navigation", "purpose": "Guide to stairs.", "input": "stairs waypoint", "output": "stairs route"},
                    {"capability": "object_detection", "purpose": "Find the exit.", "input": "new scene", "output": "exit bbox"},
                    {"capability": "navigation", "purpose": "Guide to Uber pickup.", "input": "exit waypoint", "output": "pickup route"},
                ],
            },
            supported_capabilities=["object_detection", "spatial_relationship", "navigation"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(result["stage_count"], 5)
        self.assertGreater(result["complexity_penalty"], 0)
        self.assertGreater(result["pipeline_score"], 0)

    def test_information_reduction_rejects_simple_ocr_extra_reasoning_stage(self):
        original = model_router.normalize_pipeline_analysis(
            {
                "should_chain": True,
                "reason": "Over-eager OCR plus reasoning.",
                "artifact_value_score": 0.8,
                "information_reduction": "moderate",
                "stages": [
                    {"capability": "ocr", "purpose": "Read label.", "input": "image", "output": "text"},
                    {"capability": "general_reasoning", "purpose": "Restate text.", "input": "text", "output": "answer"},
                ],
            },
            supported_capabilities=["ocr", "general_reasoning"],
        )

        result = model_router.infer_pipeline_from_information_reduction(
            "Read the medication bottle label.",
            {
                "tasks": [
                    {"name": "ocr", "weight": 0.8, "reason": "Read label."},
                    {"name": "general_reasoning", "weight": 0.2, "reason": "Answer user."},
                ],
                "latency_sensitivity": {"level": "medium", "weight": 1.0, "reason": ""},
            },
            original,
            ["ocr", "general_reasoning"],
        )

        self.assertFalse(result["should_chain"])

    def test_information_reduction_rejects_holistic_chart_pipeline(self):
        original = model_router.normalize_pipeline_analysis(
            {
                "should_chain": True,
                "reason": "Over-eager chart pipeline.",
                "artifact_value_score": 0.8,
                "information_reduction": "moderate",
                "stages": [
                    {"capability": "ocr", "purpose": "Read chart labels.", "input": "chart", "output": "text"},
                    {"capability": "map_web", "purpose": "Interpret chart.", "input": "text", "output": "answer"},
                ],
            },
            supported_capabilities=["ocr", "map_web"],
        )

        result = model_router.infer_pipeline_from_information_reduction(
            "Interpret a weather chart.",
            {
                "tasks": [
                    {"name": "ocr", "weight": 0.2, "reason": "Read labels."},
                    {"name": "map_web", "weight": 0.8, "reason": "Interpret chart."},
                ],
                "latency_sensitivity": {"level": "medium", "weight": 1.0, "reason": ""},
            },
            original,
            ["ocr", "map_web"],
        )

        self.assertFalse(result["should_chain"])

    def test_information_reduction_preserves_good_three_stage_bus_plan(self):
        original = model_router.normalize_pipeline_analysis(
            {
                "should_chain": True,
                "reason": "Bus detection, entrance localization, and guidance each preserve a useful artifact.",
                "artifact_value_score": 0.95,
                "information_reduction": "significant",
                "stages": [
                    {"capability": "object_detection", "purpose": "Find the correct bus.", "input": "scene", "output": "bus bbox"},
                    {"capability": "spatial_relationship", "purpose": "Locate entrance relative to bus.", "input": "bus bbox", "output": "entrance waypoint"},
                    {"capability": "navigation", "purpose": "Guide user to entrance.", "input": "entrance waypoint", "output": "movement guidance"},
                ],
            },
            supported_capabilities=["object_detection", "spatial_relationship", "navigation"],
        )

        result = model_router.infer_pipeline_from_information_reduction(
            "Identify the correct bus, locate the entrance, and guide me there.",
            {
                "tasks": [
                    {"name": "object_detection", "weight": 0.4, "reason": "Find bus."},
                    {"name": "spatial_relationship", "weight": 0.3, "reason": "Locate entrance."},
                    {"name": "navigation", "weight": 0.3, "reason": "Guide user."},
                ],
                "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": ""},
            },
            original,
            ["object_detection", "spatial_relationship", "navigation"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(
            [stage["capability"] for stage in result["stages"]],
            ["object_detection", "spatial_relationship", "navigation"],
        )

    def test_information_reduction_infers_three_stage_bus_entrance_from_text(self):
        result = model_router.infer_pipeline_from_information_reduction(
            "Identify the correct bus and tell me where the entrance is.",
            {
                "tasks": [
                    {"name": "object_detection", "weight": 0.75, "reason": "Find bus."},
                    {"name": "general_reasoning", "weight": 0.25, "reason": "Locate entrance."},
                ],
                "latency_sensitivity": {"level": "high", "weight": 0.8, "reason": ""},
            },
            model_router.normalize_pipeline_analysis(
                {
                    "should_chain": True,
                    "reason": "Overly generic downstream reasoning.",
                    "artifact_value_score": 0.85,
                    "information_reduction": "significant",
                    "stages": [
                        {"capability": "object_detection", "purpose": "Find bus.", "input": "scene", "output": "bus bbox"},
                        {"capability": "general_reasoning", "purpose": "Reason about entrance.", "input": "bus bbox", "output": "answer"},
                    ],
                },
                supported_capabilities=["object_detection", "general_reasoning", "spatial_relationship", "navigation"],
            ),
            ["object_detection", "general_reasoning", "spatial_relationship", "navigation"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(
            [stage["capability"] for stage in result["stages"]],
            ["object_detection", "spatial_relationship", "navigation"],
        )

    def test_information_flow_read_then_summarize_uses_ocr_then_reasoning(self):
        result = model_router.infer_pipeline_from_information_reduction(
            "Read the medication label and summarize it.",
            {
                "tasks": [
                    {"name": "ocr", "weight": 0.7, "reason": "read label text"},
                    {"name": "general_reasoning", "weight": 0.3, "reason": "summarize content"},
                ],
                "latency_sensitivity": {"level": "low", "weight": 1.0, "reason": "not urgent"},
            },
            {"should_chain": False, "reason": "Too conservative.", "stages": []},
            ["ocr", "general_reasoning", "object_detection"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(
            [stage["capability"] for stage in result["stages"]],
            ["ocr", "general_reasoning"],
        )

    def test_rank_models_rewrites_inconsistent_pipeline_by_artifact_flow(self):
        profiles = [
            model_router.ModelProfile("detector", "specialized_expert", 80, "test", {"object_detection": 1.0}, latency=0.98),
            model_router.ModelProfile("navigator", "general_vlm", 900, "test", {"navigation": 0.9, "camera_motion": 0.3}, latency=0.78),
            model_router.ModelProfile("camera_guide", "general_vlm", 700, "test", {"camera_motion": 0.9, "navigation": 0.2}, latency=0.84),
        ]
        routing_analysis = {
            "tasks": [
                {"name": "navigation", "weight": 0.7, "reason": "guidance toward target"},
                {"name": "camera_motion", "weight": 0.3, "reason": "camera movement guidance"},
            ],
            "latency_sensitivity": {"level": "high", "weight": 1.0, "reason": "live guidance"},
        }
        bad_pipeline = {
            "should_chain": True,
            "reason": "incorrect capability list as stages",
            "stages": [
                {"capability": "camera_motion", "purpose": "", "input": "", "output": "", "preferred_model_type": ""},
                {"capability": "object_detection", "purpose": "", "input": "", "output": "", "preferred_model_type": ""},
                {"capability": "spatial_relationship", "purpose": "", "input": "", "output": "", "preferred_model_type": ""},
                {"capability": "navigation", "purpose": "", "input": "", "output": "", "preferred_model_type": ""},
            ],
        }

        result = model_router.rank_models(
            "Guide my camera toward the target.",
            profiles,
            routing_analysis=routing_analysis,
            pipeline_analysis=bad_pipeline,
        )

        self.assertEqual(result["final_execution_plan"]["mode"], "pipeline")
        self.assertEqual(
            [stage["capability"] for stage in result["pipeline_analysis"]["stages"]],
            ["object_detection", "camera_motion"],
        )

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
