import unittest
from unittest.mock import patch

from model_registry import DEFAULT_TAKE_PHOTO_POLICY, MODEL_REGISTRY
from policy_executor import ToolPolicyError, execute_tool_policy, validate_tool_policy
from strategy_registry import STRATEGY_REGISTRY
from tool_policy_runtime import resolve_tool_policy
import tool_policy_runtime
import model_execution


class PolicyExecutorTests(unittest.TestCase):
    def test_registry_contains_required_models_and_strategies(self):
        self.assertTrue({"yolo", "moondream", "gemini-3.1-flash-lite", "gpt-5", "gpt-4o-mini"} <= MODEL_REGISTRY.keys())
        self.assertEqual(
            set(STRATEGY_REGISTRY),
            {"single", "cascade", "parallel_first", "parallel_aggregate", "conditional"},
        )

    def test_default_is_exactly_one_gemini_call(self):
        calls = []

        def call(model, prompt, image, metadata):
            calls.append((model, prompt, image))
            return "answer"

        result = execute_tool_policy(DEFAULT_TAKE_PHOTO_POLICY, "task", b"image", call)
        self.assertEqual(result, "answer")
        self.assertEqual(calls, [("gemini-3.1-flash-lite", "task", b"image")])

    def test_cascade_uses_evaluator_and_fallback(self):
        calls = []

        def call(model, prompt, image, metadata):
            calls.append(model)
            if model == "gpt-4o-mini":
                return "YES" if "second" in prompt else "NO"
            return "first" if model == "moondream" else "second"

        result = execute_tool_policy({
            "strategy": "cascade",
            "models": ["moondream", "gemini-3.1-flash-lite"],
            "evaluator": "gpt-4o-mini",
            "stop_condition": "accepted",
        }, "task", b"image", call)
        self.assertEqual(result, "second")
        self.assertEqual(calls, ["moondream", "gpt-4o-mini", "gemini-3.1-flash-lite", "gpt-4o-mini"])

    def test_parallel_aggregate_uses_aggregator(self):
        def call(model, prompt, image, metadata):
            return "combined" if model == "gpt-4o-mini" else model

        result = execute_tool_policy({
            "strategy": "parallel_aggregate",
            "models": ["moondream", "gemini-3.1-flash-lite"],
            "aggregator": "gpt-4o-mini",
        }, "task", b"image", call)
        self.assertEqual(result, "combined")

    def test_conditional_uses_explicit_metadata_only(self):
        calls = []

        def call(model, prompt, image, metadata):
            calls.append(model)
            return model

        policy = {
            "strategy": "conditional",
            "condition": {"key": "needs_accuracy", "equals": True},
            "if_true": {"strategy": "single", "models": ["gpt-5"]},
            "if_false": {"strategy": "single", "models": ["gemini-3.1-flash-lite"]},
        }
        self.assertEqual(execute_tool_policy(policy, "task", b"image", call, metadata={"needs_accuracy": True}), "gpt-5")
        self.assertEqual(calls, ["gpt-5"])

    def test_rejects_unknown_models(self):
        with self.assertRaises(ToolPolicyError):
            validate_tool_policy({"strategy": "single", "models": ["unknown"]})

    def test_invalid_explicit_policy_falls_back_to_default(self):
        self.assertEqual(
            resolve_tool_policy({"strategy": "not-supported", "models": ["gpt-5"]}),
            DEFAULT_TAKE_PHOTO_POLICY,
        )

    def test_rejects_unsupported_fields(self):
        with self.assertRaises(ToolPolicyError):
            validate_tool_policy({
                "strategy": "single",
                "models": ["gemini-3.1-flash-lite"],
                "custom_router": True,
            })

    def test_runtime_delegates_explicit_policy_only_to_policy_executor(self):
        explicit = {"strategy": "single", "models": ["gpt-5"]}
        with patch.object(tool_policy_runtime, "execute_tool_policy", return_value="ok") as execute:
            result = tool_policy_runtime.execute_resolved_tool_policy(
                "task", b"image", policy=explicit, request_id="test"
            )
        self.assertEqual(result, "ok")
        self.assertEqual(execute.call_args.args[0], explicit)

    def test_runtime_sends_default_to_policy_executor_when_policy_is_invalid(self):
        with patch.object(tool_policy_runtime, "execute_tool_policy", return_value="ok") as execute:
            tool_policy_runtime.execute_resolved_tool_policy(
                "task", b"image", policy={"strategy": "bad"}, request_id="test"
            )
        self.assertEqual(execute.call_args.args[0], DEFAULT_TAKE_PHOTO_POLICY)

    def test_public_executor_preserves_exact_declared_policy(self):
        explicit = {"strategy": "parallel_first", "models": ["moondream", "gpt-5"]}
        with patch.object(model_execution, "execute_resolved_tool_policy", return_value="ok") as execute:
            result = model_execution.execute_tool_policy(
                image=b"image", prompt="task", policy=explicit, tool_name="test"
            )
        self.assertEqual(result, "ok")
        self.assertIs(execute.call_args.kwargs["policy"], explicit)
        self.assertEqual(execute.call_args.kwargs["metadata"]["tool_name"], "test")

    def test_policy_logs_declared_strategy_and_models(self):
        with self.assertLogs("policy_executor", level="INFO") as captured:
            execute_tool_policy(
                {"strategy": "single", "models": ["gemini-3.1-flash-lite"]},
                "task", b"image", lambda model, prompt, image, metadata: "answer",
            )
        output = "\n".join(captured.output)
        self.assertIn("strategy=single", output)
        self.assertIn("gemini-3.1-flash-lite", output)

    def test_public_executor_missing_policy_uses_one_gemini_call(self):
        calls = []
        with patch.object(tool_policy_runtime, "IMPLEMENTATION_EXECUTORS", {
            "model": lambda profile, messages, images, metadata: type(
                "Result", (), {"response": "answer"}
            )(),
        }), patch.object(tool_policy_runtime, "_response_text", side_effect=lambda value: calls.append(value) or value):
            result = model_execution.execute_tool_policy(image=b"image", prompt="task")
        self.assertEqual(result, "answer")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
