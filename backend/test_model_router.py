"""Offline regression tests for atomic capability execution."""

from pathlib import Path
import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_router


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class TestGoogleVisionExecutor(unittest.TestCase):
    def test_uses_adc_oauth_token_for_rest_request(self):
        class Credentials:
            valid = False
            token = None
            project_id = "credential-project"

            def refresh(self, request):
                self.refresh_request = request
                self.valid = True
                self.token = "oauth-access-token"

        class HttpResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps({
                    "responses": [{"textAnnotations": [{"description": "Hello document"}]}]
                }).encode("utf-8")

        credentials = Credentials()
        captured = {}

        def urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return HttpResponse()

        profile = model_router.ImplementationProfile("google_vision", "google_vision")
        with patch("google.auth.default", return_value=(credentials, "detected-project")) as default, \
             patch.object(model_router.urllib.request, "urlopen", side_effect=urlopen), \
             patch.dict(model_router.os.environ, {
                 "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/service-account.json",
                 "GEMINI_API_KEY": "must-not-be-used-for-vision",
             }), self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router._google_vision_executor(
                profile, [], [b"image-bytes"], {"timeout": 12}
            )

        default.assert_called_once_with(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.assertTrue(credentials.valid)
        self.assertEqual(captured["request"].full_url, "https://vision.googleapis.com/v1/images:annotate")
        self.assertEqual(captured["request"].headers["Authorization"], "Bearer oauth-access-token")
        self.assertNotIn("must-not-be-used-for-vision", captured["request"].full_url)
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(model_router._response_text(result.response), "Hello document")
        self.assertEqual(result.artifact, {"text": "Hello document", "accepted": True})
        output = "\n".join(logs.output)
        self.assertIn("method=application_default_credentials_oauth_rest", output)
        self.assertIn("default_credentials_loaded=true", output)
        self.assertIn("project_id=detected-project", output)
        self.assertIn("vision_client_created=false reason=direct_rest_provider", output)
        self.assertEqual(captured["request"].headers["X-goog-user-project"], "detected-project")

    def test_logs_google_error_response_body(self):
        class Credentials:
            valid = True
            token = "oauth-access-token"
            project_id = "programat"
            service_account_email = "programat@programat.iam.gserviceaccount.com"

        error_body = b'{"error":{"code":403,"message":"SERVICE_DISABLED"}}'
        http_error = model_router.urllib.error.HTTPError(
            "https://vision.googleapis.com/v1/images:annotate",
            403,
            "Forbidden",
            {},
            io.BytesIO(error_body),
        )
        profile = model_router.ImplementationProfile("google_vision", "google_vision")
        with patch("google.auth.default", return_value=(Credentials(), "programat")), \
             patch.object(model_router.urllib.request, "urlopen", side_effect=http_error), \
             self.assertLogs(model_router.logger, level="INFO") as logs, \
             self.assertRaisesRegex(RuntimeError, "SERVICE_DISABLED"):
            model_router._google_vision_executor(profile, [], [b"image"], {})

        output = "\n".join(logs.output)
        self.assertIn("endpoint=https://vision.googleapis.com/v1/images:annotate", output)
        self.assertIn("status=403", output)
        self.assertIn('response_body={"error":{"code":403,"message":"SERVICE_DISABLED"}}', output)


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
        self.assertTrue(config["planner_enabled"])
        self.assertTrue(config["routing_enabled"])
        self.assertEqual(config["system_model"], "llama_planner")
        self.assertEqual(config["default_llm_when_routing_disabled"], "gemini_flash_lite")

    def test_planner_disabled_forces_routing_disabled_with_warning(self):
        config = {
            "global": {
                "planner_enabled": False,
                "routing_enabled": True,
                "system_model": {"implementation": "planner"},
                "default_llm_when_routing_disabled": {"implementation": "default"},
            },
        }
        with patch.object(model_router, "_load_yaml", return_value=config), \
             self.assertLogs(model_router.logger, level="WARNING") as logs:
            loaded = model_router.load_global_execution_config(Path("unused.yaml"))

        self.assertFalse(loaded["planner_enabled"])
        self.assertFalse(loaded["routing_enabled"])
        self.assertIn("forcing routing_enabled=false", "\n".join(logs.output))

    def test_deprecated_llava_is_not_configured_or_selected(self):
        self.assertNotIn("llava", model_router.load_implementation_profiles())
        policies = model_router.load_execution_policies()
        for policy in policies.values():
            self.assertNotIn("llava", policy["candidates"])

    def test_reasoning_cascade_uses_gemini_then_gpt4o(self):
        policy = model_router.load_execution_policies()["general_reasoning"]
        self.assertEqual(
            policy["candidates"],
            ["gemini_flash_lite", "gpt4o"],
        )
        self.assertEqual(policy["evaluator"], "gpt4o-mini")

class TestAtomicCopilotCall(unittest.TestCase):
    def test_planner_disabled_bypasses_policies_and_calls_default_once(self):
        profiles = {
            "default": model_router.ImplementationProfile("default", "model", model="test/default"),
            "system": model_router.ImplementationProfile("system", "model", model="test/system"),
        }
        config = {
            "planner_enabled": False,
            "routing_enabled": False,
            "system_model": "system",
            "default_llm_when_routing_disabled": "default",
        }
        with patch.object(model_router, "load_global_execution_config", return_value=config), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.object(model_router, "load_execution_policies") as policies, \
             patch.object(model_router, "call_model", return_value=response("Direct answer")) as call_model, \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(
                capability="ocr", goal="Read this document", images=[b"original-image"]
            )

        policies.assert_not_called()
        call_model.assert_called_once()
        self.assertEqual(call_model.call_args.args[0], "test/default")
        self.assertEqual(call_model.call_args.kwargs["images"], [b"original-image"])
        self.assertEqual(result["response"], "Direct answer")
        self.assertIn("planner disabled -> single-stage execution", "\n".join(logs.output))

    def test_tool_execution_image_is_used_when_stage_omits_images(self):
        seen_images = []

        def executor(_profile, _messages, images, _metadata):
            seen_images.extend(images)
            return model_router.ImplementationResult(response("Independent inspection"))

        policies = {"general_reasoning": {
            "candidates": ["vision"], "evaluator": None, "cascade": None,
        }}
        profiles = {"vision": model_router.ImplementationProfile("vision", "fake")}
        token = model_router.TOOL_EXECUTION_IMAGES.set([b"current-original-image"])
        try:
            with patch.object(model_router, "load_execution_policies", return_value=policies), \
                 patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
                 patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
                model_router.copilot_llm_call(capability="general_reasoning")
        finally:
            model_router.TOOL_EXECUTION_IMAGES.reset(token)

        self.assertEqual(seen_images, [b"current-original-image"])

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
        self.assertIn("additional context", calls[0][0]["content"])
        self.assertIn("not evidence that the target is absent", calls[0][0]["content"])
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

        self.assertEqual(result["response"], "The previous stage could not produce a reliable result.")
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
        self.assertIn("previous-stage artifact", calls[0][1][-2]["content"])
        evaluator_prompt = calls[1][1][0]["content"]
        self.assertIn('Target labels, when relevant: ["exit", "door", "doorway", "exit sign"]', evaluator_prompt)
        self.assertIn("enough useful information", evaluator_prompt)
        self.assertIn("Navigation requires actionable guidance", evaluator_prompt)
        self.assertIn("Do not answer NO merely", evaluator_prompt)
        self.assertIn("Output only YES or NO", evaluator_prompt)
        self.assertEqual(calls[1][2], [])

    def test_exit_target_is_inferred_and_evaluator_receives_grounding_context(self):
        calls = []

        def executor(profile, messages, images, metadata):
            calls.append((profile.name, messages, images, metadata))
            return model_router.ImplementationResult(response("YES" if profile.name == "judge" else "The exit is to your right."))

        policies = {"navigation": {
            "candidates": ["nav", "fallback"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {name: model_router.ImplementationProfile(name, "fake") for name in ("nav", "fallback", "judge")}
        artifact = {"detections": [{"label": "door", "location": "right"}, {"label": "toilet", "location": "left"}]}
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(
                capability="navigation", goal="Guide me to the exit",
                metadata={"previous_stage_artifact": artifact},
            )

        self.assertEqual(result["response"], "The exit is to your right.")
        nav_metadata = calls[0][3]
        self.assertEqual(nav_metadata["target_labels"], model_router.EXIT_TARGET_LABELS)
        self.assertEqual([item["label"] for item in nav_metadata["previous_stage_artifact"]["detections"]], ["door"])
        prompt = calls[1][1][0]["content"]
        self.assertIn('Target labels, when relevant: ["exit", "door", "doorway", "exit sign"]', prompt)
        self.assertIn('"label": "door"', prompt)
        self.assertNotIn('"label": "toilet"', prompt)

    def test_navigation_independently_inspects_image_when_artifact_has_no_target(self):
        calls = []

        def executor(_profile, messages, images, _metadata):
            calls.append((messages, images))
            return model_router.ImplementationResult(response("The exit is visible on the left."))

        policies = {"navigation": {"candidates": ["nav"], "evaluator": None, "cascade": None}}
        profiles = {"nav": model_router.ImplementationProfile("nav", "fake")}
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(
                capability="navigation", goal="Navigate to the exit",
                metadata={"previous_stage_artifact": {"detections": [{"label": "toilet", "location": "right"}]}},
                images=[b"original-image"],
            )

        self.assertEqual(result["response"], "The exit is visible on the left.")
        self.assertEqual(calls[0][1], [b"original-image"])
        prompt_text = "\n".join(message["content"] for message in calls[0][0])
        self.assertNotIn("previous-stage artifact", prompt_text)
        self.assertNotIn("toilet", prompt_text)

    def test_legacy_unrelated_detection_text_is_uncertainty_not_absence(self):
        calls = []

        def executor(_profile, messages, images, _metadata):
            calls.append((messages, images))
            return model_router.ImplementationResult(response("I can see an exit ahead."))

        policies = {"navigation": {"candidates": ["nav"], "evaluator": None, "cascade": None}}
        profiles = {"nav": model_router.ImplementationProfile("nav", "fake")}
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(
                capability="navigation", goal="Navigate to the nearest exit",
                metadata={"previous_stage_output": "toilet at right"}, images=[b"image"],
            )

        self.assertEqual(result["response"], "I can see an exit ahead.")
        self.assertEqual(calls[0][1], [b"image"])
        self.assertNotIn("toilet", "\n".join(message["content"] for message in calls[0][0]))

    def test_empty_and_explicitly_low_confidence_artifacts_are_not_useful(self):
        self.assertFalse(model_router._artifact_is_useful({"detections": []}))
        self.assertFalse(model_router._artifact_is_useful({"text": ""}))
        self.assertFalse(model_router._artifact_is_useful({"text": "Stage failed", "accepted": False}))
        self.assertFalse(model_router._artifact_is_useful({"text": "possible label", "low_confidence": True}))
        self.assertFalse(model_router._artifact_is_useful({"text": "possible label", "confidence": "low"}))
        self.assertTrue(model_router._artifact_is_useful({"text": "Take one tablet daily"}))
        self.assertTrue(model_router._artifact_is_useful({"detections": [{"label": "document"}]}))

    def test_target_filter_keeps_exit_aliases_only(self):
        artifact = model_router._filter_target_artifact(
            {"detections": [{"label": "door"}, {"label": "TV"}, {"label": "exit sign"}]},
            model_router.EXIT_TARGET_LABELS,
        )
        self.assertEqual([item["label"] for item in artifact["detections"]], ["door", "exit sign"])
        self.assertTrue(artifact["matching_detection"])

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
            is_evaluation = "Output only YES or NO" in messages[0].get("content", "")
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
            is_evaluation = "Output only YES or NO" in messages[0].get("content", "")
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
            is_evaluation = "Output only YES or NO" in messages[0].get("content", "")
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
            "planner_enabled": True,
            "routing_enabled": False,
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
        self.assertIn("planner_enabled=true routing_enabled=false", output)
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
            "planner_enabled": True,
            "routing_enabled": False,
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
            "planner_enabled": False,
            "routing_enabled": False,
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
