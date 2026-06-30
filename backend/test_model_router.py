"""Offline regression tests for atomic capability execution."""

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_router


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class TestExecutionPolicyConfiguration(unittest.TestCase):
    def test_policy_covers_taxonomy_and_cascades_only_reasoning_capabilities(self):
        capabilities = set(model_router.load_capability_profiles())
        policies = model_router.load_execution_policies()
        self.assertEqual(set(policies), capabilities)
        self.assertEqual(policies["object_detection_localization"]["candidates"], ["yolo"])
        self.assertIsNone(policies["object_detection_localization"]["evaluator"])
        self.assertEqual(policies["ocr"]["candidates"], ["google_vision"])
        default_reasoning = policies["navigation"]
        self.assertTrue(default_reasoning["candidates"])
        self.assertTrue(default_reasoning["evaluator"])
        self.assertEqual(default_reasoning["cascade"], "default_reasoning")
        for capability in capabilities - {"object_detection_localization", "ocr"}:
            self.assertEqual(policies[capability], default_reasoning)

    def test_first_implementation_exists(self):
        model_router.validate_execution_configuration()

    def test_policy_loader_rejects_unknown_taxonomy(self):
        with patch.object(
            model_router,
            "_load_yaml",
            return_value={"not_a_capability": {"implementation": "fake"}},
        ), patch.object(
            model_router,
            "load_capability_profiles",
            return_value={"ocr": {}},
        ), self.assertRaises(model_router.ExecutionPolicyError):
            model_router.load_execution_policies(Path("unused.yaml"))

    def test_cascade_candidate_order_and_evaluator_come_only_from_yaml(self):
        config = {
            "cascade_profiles": {
                "custom": {"candidates": ["small", "large"], "evaluator": "judge"},
            },
            "navigation": {"cascade": "custom"},
        }
        with patch.object(model_router, "_load_yaml", return_value=config), \
             patch.object(model_router, "load_capability_profiles", return_value={"navigation": {}}):
            policies = model_router.load_execution_policies(Path("unused.yaml"))

        self.assertEqual(policies["navigation"], {
            "candidates": ["small", "large"],
            "evaluator": "judge",
            "cascade": "custom",
            "specialized": False,
        })

    def test_global_switch_and_models_load_from_execution_policy(self):
        config = model_router.load_global_execution_config()
        self.assertIsInstance(config["model_routing_enabled"], bool)
        self.assertEqual(config["system_model"], "llama_planner")
        self.assertEqual(config["default_llm_when_routing_disabled"], "gemini_flash_lite")


class TestAtomicCopilotCall(unittest.TestCase):
    def test_uses_only_first_implementation_and_returns_structured_result(self):
        calls = []

        def executor(profile, messages, images, metadata):
            calls.append((profile.name, messages, images, metadata))
            return model_router.ImplementationResult(
                response("Exit at the right"),
                {"detections": [{"label": "exit", "bbox": [1, 2, 3, 4]}]},
            )

        policies = {"object_detection_localization": {
            "candidates": ["first", "unused"], "evaluator": None, "cascade": None,
        }}
        profiles = {
            "first": model_router.ImplementationProfile("first", "fake"),
            "unused": model_router.ImplementationProfile("unused", "fake"),
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(
                capability="object_detection_localization",
                messages=[{"role": "user", "content": "Locate the exit"}],
                images=[b"image"],
            )

        self.assertEqual([call[0] for call in calls], ["first"])
        self.assertEqual(result, {
            "response": "Exit at the right",
            "artifact": {"detections": [{"label": "exit", "bbox": [1, 2, 3, 4]}]},
            "implementation": "first",
            "capability": "object_detection_localization",
        })

    def test_response_becomes_default_text_artifact(self):
        calls = []

        def executor(_profile, messages, _images, _metadata):
            calls.append(messages)
            return model_router.ImplementationResult(response("Turn left"))

        with patch.object(model_router, "load_execution_policies", return_value={"navigation": {
                 "candidates": ["nav"], "evaluator": None, "cascade": None,
             }}), \
             patch.object(model_router, "load_implementation_profiles", return_value={
                 "nav": model_router.ImplementationProfile("nav", "fake")
             }), patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="navigation")

        self.assertEqual(result["artifact"], {"text": "Turn left"})
        self.assertEqual(calls[0][0]["role"], "system")
        self.assertIn("blind or low-vision", calls[0][0]["content"])
        self.assertIn("primary source of truth", calls[0][0]["content"])
        self.assertIn("at most 2 short sentences", calls[0][0]["content"])

    def test_unknown_capability_is_rejected(self):
        with patch.object(model_router, "load_execution_policies", return_value={"ocr": {
                 "candidates": ["reader"], "evaluator": None, "cascade": None,
             }}), \
             patch.object(model_router, "load_implementation_profiles", return_value={}):
            with self.assertRaises(model_router.ExecutionPolicyError):
                model_router.copilot_llm_call(capability="unknown")

    def test_legacy_object_detection_name_is_normalized(self):
        def executor(*_args):
            return model_router.ImplementationResult(response("Exit found"), {"detections": []})

        policies = {"object_detection_localization": {
            "candidates": ["detector"], "evaluator": None, "cascade": None,
        }}
        profiles = {"detector": model_router.ImplementationProfile("detector", "fake")}
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="object_detection")

        self.assertEqual(result["capability"], "object_detection_localization")
        self.assertEqual(result["implementation"], "detector")

    def test_legacy_reasoning_names_normalize_to_current_taxonomy(self):
        self.assertEqual(model_router.normalize_capability_name("spatial_relationship"), "spatial_reasoning")
        self.assertEqual(model_router.normalize_capability_name("map_web"), "structured_visual_understanding")
        self.assertEqual(model_router.normalize_capability_name("video"), "temporal_reasoning")

    def test_fixed_implementation_error_returns_user_fallback(self):
        def executor(*_args):
            raise RuntimeError("failed once")

        with patch.object(model_router, "load_execution_policies", return_value={"ocr": {
                 "candidates": ["reader", "fallback"], "evaluator": None, "cascade": None,
             }}), \
             patch.object(model_router, "load_implementation_profiles", return_value={
                 "reader": model_router.ImplementationProfile("reader", "fake"),
                 "fallback": model_router.ImplementationProfile("fallback", "fake"),
             }), patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="ocr")

        self.assertEqual(result["response"], "Sorry, I couldn't generate guidance from this image.")
        self.assertEqual(result["implementation"], "fallback")

    def test_reasoning_returns_llava_response_when_evaluator_says_yes(self):
        calls = []

        def executor(profile, messages, images, metadata):
            calls.append((profile.name, messages, images, metadata))
            text = "YES" if profile.name == "gpt4o" else "Turn right toward the exit."
            return model_router.ImplementationResult(response(text))

        policies = {"navigation": {
            "candidates": ["llava", "gpt4o"], "evaluator": "gpt4o", "cascade": "test",
        }}
        profiles = {
            "llava": model_router.ImplementationProfile("llava", "fake"),
            "gpt4o": model_router.ImplementationProfile("gpt4o", "fake"),
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(
                capability="navigation",
                goal="Guide me to the exit",
                metadata={"previous_stage_artifact": {"detections": [{"label": "exit"}]}},
            )

        self.assertEqual([call[0] for call in calls], ["llava", "gpt4o"])
        self.assertEqual(result["response"], "Turn right toward the exit.")
        self.assertEqual(result["implementation"], "llava")
        self.assertIn("Previous-stage artifact", calls[0][1][-2]["content"])
        self.assertIn("Answer only YES or NO", calls[1][1][0]["content"])
        self.assertEqual(calls[1][2], [])

    def test_reasoning_escalates_to_gpt4o_when_evaluator_says_no(self):
        gpt4o_calls = 0

        def executor(profile, messages, _images, _metadata):
            nonlocal gpt4o_calls
            if profile.name == "llava":
                return model_router.ImplementationResult(response("A TV is ahead."))
            gpt4o_calls += 1
            if gpt4o_calls == 1:
                return model_router.ImplementationResult(response("NO"))
            self.assertIn("blind or low-vision", messages[0]["content"].lower())
            return model_router.ImplementationResult(response("The exit is to your right."))

        policies = {"navigation": {
            "candidates": ["llava", "gpt4o"], "evaluator": "gpt4o", "cascade": "test",
        }}
        profiles = {
            "llava": model_router.ImplementationProfile("llava", "fake"),
            "gpt4o": model_router.ImplementationProfile("gpt4o", "fake"),
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="navigation")

        self.assertEqual(gpt4o_calls, 2)
        self.assertEqual(result["response"], "The exit is to your right.")
        self.assertEqual(result["implementation"], "gpt4o")

    def test_cascade_logs_each_candidate_and_yes_no_only(self):
        decisions = iter(["NO", "NO"])

        def executor(profile, messages, _images, _metadata):
            is_evaluation = "Answer only YES or NO" in messages[0].get("content", "")
            if profile.name == "gpt4o" and is_evaluation:
                return model_router.ImplementationResult(response(next(decisions)))
            return model_router.ImplementationResult(response(f"response from {profile.name}"))

        policies = {"navigation": {
            "candidates": ["llava", "qwen", "gpt4o"],
            "evaluator": "gpt4o",
            "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("llava", "qwen", "gpt4o")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(capability="navigation")

        output = "\n".join(logs.output)
        self.assertIn("implementation=llava response:\nresponse from llava", output)
        self.assertIn("implementation=qwen response:\nresponse from qwen", output)
        self.assertIn("implementation=gpt4o response:\nresponse from gpt4o", output)
        self.assertEqual(output.count("evaluator=gpt4o decision=NO"), 2)
        self.assertIn("selected implementation=gpt4o", output)
        self.assertEqual(result["response"], "response from gpt4o")

    def test_returns_best_response_when_later_candidate_fails(self):
        def executor(profile, messages, _images, _metadata):
            is_evaluation = "Answer only YES or NO" in messages[0].get("content", "")
            if profile.name == "judge" and is_evaluation:
                return model_router.ImplementationResult(response("NO"))
            if profile.name == "llava":
                return model_router.ImplementationResult(response("Turn slightly right."))
            raise RuntimeError("provider unavailable")

        policies = {"navigation": {
            "candidates": ["llava", "gpt4o"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("llava", "gpt4o", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(capability="navigation")

        output = "\n".join(logs.output)
        self.assertEqual(result["response"], "Turn slightly right.")
        self.assertEqual(result["implementation"], "llava")
        self.assertIn("implementation=gpt4o failed=provider unavailable", output)
        self.assertIn("fallback=llava", output)

    def test_evaluator_failure_continues_to_next_candidate(self):
        def executor(profile, messages, _images, _metadata):
            is_evaluation = "Answer only YES or NO" in messages[0].get("content", "")
            if profile.name == "judge" and is_evaluation:
                raise TimeoutError("evaluation timed out")
            return model_router.ImplementationResult(response(f"response from {profile.name}"))

        policies = {"navigation": {
            "candidates": ["llava", "qwen"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("llava", "qwen", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(capability="navigation")

        self.assertEqual(result["response"], "response from qwen")
        self.assertIn("evaluator=judge decision=FAILED", "\n".join(logs.output))

    def test_empty_response_is_skipped(self):
        def executor(profile, _messages, _images, _metadata):
            text = "" if profile.name == "empty" else "usable response"
            return model_router.ImplementationResult(response(text))

        policies = {"navigation": {
            "candidates": ["empty", "usable"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("empty", "usable", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="navigation")

        self.assertEqual(result["response"], "usable response")
        self.assertEqual(result["implementation"], "usable")

    def test_routing_disabled_uses_default_for_non_specialized_capability(self):
        calls = []

        def executor(profile, _messages, _images, _metadata):
            calls.append(profile.name)
            return model_router.ImplementationResult(response("default response"))

        policies = {"navigation": {
            "candidates": ["llava", "gpt4o"],
            "evaluator": "gpt4o",
            "cascade": "reasoning",
            "specialized": False,
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake", model=f"test/{name}")
            for name in ("llava", "gpt4o", "default", "system")
        }
        global_config = {
            "model_routing_enabled": False,
            "system_model": "system",
            "default_llm_when_routing_disabled": "default",
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.object(model_router, "load_global_execution_config", return_value=global_config), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(capability="navigation")

        self.assertEqual(calls, ["default"])
        self.assertEqual(result["implementation"], "default")
        output = "\n".join(logs.output)
        self.assertIn("model_routing_enabled=false", output)
        self.assertIn("system_model=system/test/system", output)
        self.assertIn("default_llm_when_routing_disabled=default/test/default", output)
        self.assertIn("capability=navigation selected implementation=default", output)

    def test_routing_disabled_keeps_specialized_capability(self):
        calls = []

        def executor(profile, _messages, _images, _metadata):
            calls.append(profile.name)
            return model_router.ImplementationResult(response("detected"))

        policies = {"object_detection_localization": {
            "candidates": ["yolo"],
            "evaluator": None,
            "cascade": None,
            "specialized": True,
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake", model=f"test/{name}")
            for name in ("yolo", "default", "system")
        }
        global_config = {
            "model_routing_enabled": False,
            "system_model": "system",
            "default_llm_when_routing_disabled": "default",
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.object(model_router, "load_global_execution_config", return_value=global_config), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="object_detection_localization")

        self.assertEqual(calls, ["yolo"])
        self.assertEqual(result["implementation"], "yolo")


class TestSystemCall(unittest.TestCase):
    def test_system_llm_call_uses_yaml_global_implementation(self):
        global_config = {
            "model_routing_enabled": False,
            "system_model": "planner",
            "default_llm_when_routing_disabled": "default",
        }
        profiles = {
            "planner": model_router.ImplementationProfile("planner", "model", model="test/planner"),
            "default": model_router.ImplementationProfile("default", "model", model="test/default"),
        }
        with patch.object(model_router, "load_global_execution_config", return_value=global_config), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.object(model_router, "call_model", return_value=response("ok")) as call_model, \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            model_router.system_llm_call(messages=[{"role": "user", "content": "Plan this"}])
        self.assertEqual(call_model.call_args.args[0], "test/planner")
        self.assertIn(
            "capability=system selected implementation=planner",
            "\n".join(logs.output),
        )


if __name__ == "__main__":
    unittest.main()
