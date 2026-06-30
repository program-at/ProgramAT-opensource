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
    def test_policy_covers_capability_taxonomy_with_one_implementation_each(self):
        capabilities = set(model_router.load_capability_profiles())
        policies = model_router.load_execution_policies()
        self.assertEqual(set(policies), capabilities)
        self.assertTrue(all(len(implementations) == 1 for implementations in policies.values()))
        self.assertEqual(policies["object_detection_localization"], ["yolo"])
        self.assertEqual(policies["ocr"], ["google_vision"])

    def test_first_implementation_exists(self):
        model_router.validate_execution_configuration()

    def test_policy_loader_rejects_unknown_taxonomy(self):
        with patch.object(
            model_router,
            "_load_yaml",
            return_value={"not_a_capability": {"implementations": ["fake"]}},
        ), patch.object(
            model_router,
            "load_capability_profiles",
            return_value={"ocr": {}},
        ), self.assertRaises(model_router.ExecutionPolicyError):
            model_router.load_execution_policies(Path("unused.yaml"))


class TestAtomicCopilotCall(unittest.TestCase):
    def test_uses_only_first_implementation_and_returns_structured_result(self):
        calls = []

        def executor(profile, messages, images, metadata):
            calls.append((profile.name, messages, images, metadata))
            return model_router.ImplementationResult(
                response("Exit at the right"),
                {"detections": [{"label": "exit", "bbox": [1, 2, 3, 4]}]},
            )

        policies = {"object_detection_localization": ["first", "unused"]}
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
        def executor(*_args):
            return model_router.ImplementationResult(response("Turn left"))

        with patch.object(model_router, "load_execution_policies", return_value={"navigation": ["nav"]}), \
             patch.object(model_router, "load_implementation_profiles", return_value={
                 "nav": model_router.ImplementationProfile("nav", "fake")
             }), patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="navigation")

        self.assertEqual(result["artifact"], {"text": "Turn left"})

    def test_unknown_capability_is_rejected(self):
        with patch.object(model_router, "load_execution_policies", return_value={"ocr": ["reader"]}), \
             patch.object(model_router, "load_implementation_profiles", return_value={}):
            with self.assertRaises(model_router.ExecutionPolicyError):
                model_router.copilot_llm_call(capability="unknown")

    def test_implementation_error_is_not_retried(self):
        def executor(*_args):
            raise RuntimeError("failed once")

        with patch.object(model_router, "load_execution_policies", return_value={"ocr": ["reader", "fallback"]}), \
             patch.object(model_router, "load_implementation_profiles", return_value={
                 "reader": model_router.ImplementationProfile("reader", "fake"),
                 "fallback": model_router.ImplementationProfile("fallback", "fake"),
             }), patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            with self.assertRaisesRegex(RuntimeError, "failed once"):
                model_router.copilot_llm_call(capability="ocr")


class TestSystemCall(unittest.TestCase):
    def test_system_llm_call_stays_fixed(self):
        with patch.object(model_router, "call_model", return_value=response("ok")) as call_model:
            model_router.system_llm_call(messages=[{"role": "user", "content": "Plan this"}])
        self.assertEqual(call_model.call_args.args[0], model_router.SYSTEM_MODEL)


if __name__ == "__main__":
    unittest.main()
