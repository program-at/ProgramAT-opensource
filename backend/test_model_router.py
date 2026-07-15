"""Offline regression tests for atomic capability execution."""

from pathlib import Path
import base64
import io
import json
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_router


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class TestMoondreamCloudExecutor(unittest.TestCase):
    @staticmethod
    def image_bytes(image_format="JPEG"):
        output = io.BytesIO()
        Image.new("RGBA" if image_format == "PNG" else "RGB", (40, 20), "white").save(
            output, format=image_format
        )
        return output.getvalue()

    def setUp(self):
        model_router._MOONDREAM_COOLDOWN_UNTIL = 0.0
        model_router._MOONDREAM_CONSECUTIVE_FAILURES = 0

    def profile(self):
        return model_router.ImplementationProfile(
            "moondream_cloud", "moondream_cloud", model="moondream/moondream3-preview"
        )

    def test_builds_expected_payload_and_extracts_choices_message_content(self):
        create = Mock(return_value=response("A small white image."))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(model_router, "_create_moondream_client", return_value=client) as factory, \
             patch.dict(model_router.os.environ, {
                 "MOONDREAM_API_KEY": "test-key",
                 "MOONDREAM_MODEL": "custom/moondream",
                 "MOONDREAM_TIMEOUT_SECONDS": "2.5",
                 "ENABLE_MOONDREAM_CLOUD": "true",
             }), self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router._moondream_cloud_executor(
                self.profile(),
                [
                    {"role": "system", "content": "Return only concise audio text. " * 30},
                    {"role": "user", "content": "Full ProgramAT routing and schema instructions " * 20},
                ],
                [self.image_bytes("PNG")],
                {"capability": "general_reasoning", "goal": "Which envelope should I open first?"},
            )

        factory.assert_called_once_with("test-key", 2.5)
        request = create.call_args.kwargs
        self.assertEqual(request["model"], "custom/moondream")
        self.assertEqual(len(request["messages"]), 1)
        content = request["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("Which envelope should I open first?", content[0]["text"])
        self.assertNotIn("routing and schema", content[0]["text"])
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(model_router._response_text(result.response), "A small white image.")
        output = "\n".join(logs.output)
        self.assertIn("base_url=https://api.moondream.ai/v1", output)
        self.assertIn("message_format=nested_image_url_url", output)
        self.assertIn("prompt_adapter=extractive_short", output)
        self.assertIn("prompt_source=user_goal", output)
        self.assertIn("prompt_chars_original=35", output)
        self.assertIn("prompt_chars_sent=", output)
        self.assertIn("response status=200", output)
        self.assertIn("latency_ms=", output)

    def test_prompt_adapter_normalizes_and_caps_goal(self):
        full = "```json\n# ProgramAT\n" + ("Find the nearest exit safely using visible landmarks. " * 20)
        prompt, short_goal, original_chars, source = model_router.moondream_provider.adapt_task_prompt(
            [{"role": "system", "content": "Return only concise output." * 20}],
            {"capability": "navigation", "stage_goal": full},
        )
        self.assertIn("nearest exit", prompt)
        self.assertNotIn("```", prompt)
        self.assertTrue(prompt.endswith("landmarks."))
        self.assertLessEqual(len(short_goal), 160)
        self.assertLessEqual(len(prompt), 250)
        self.assertGreater(original_chars, 100)
        self.assertEqual(source, "stage_goal")

    def test_card_goal_uses_strict_rank_suit_prompt(self):
        prompt, short_goal, _original_chars, source = model_router.moondream_provider.adapt_task_prompt(
            [{"role": "user", "content": "Identify the cards and their properties"}],
            {
                "capability": "structured_visual_understanding",
                "stage_goal": "Identify the cards and their properties",
            },
        )

        self.assertEqual(short_goal, "Identify the cards and their properties")
        self.assertLessEqual(len(prompt), 250)
        self.assertEqual(source, "stage_goal")
        self.assertEqual(
            prompt,
            model_router.moondream_provider.CARD_IDENTIFICATION_PROMPT,
        )
        self.assertNotIn("Ace of Spades", prompt)
        self.assertNotIn("Ace of Hearts", prompt)
        self.assertIn("Read only the top-left index of each physical card", prompt)
        self.assertIn("Ignore all other corners", prompt)
        self.assertNotIn("one clear corner index", prompt)
        self.assertIn("Multiple cards: left to right", prompt)
        self.assertIn("Rank+suit only", prompt)

    def test_card_detector_matches_runtime_goal_and_avoids_false_positives(self):
        true_cases = [
            "Identify the cards and their properties",
            "Card identifier",
            "Cards visible in my hand",
            "Tell me the suits and ranks",
            "rank and suit",
            "playing cards",
            "card game",
        ]
        for text in true_cases:
            with self.subTest(text=text):
                matched, _reason = model_router.moondream_provider.card_mode_reason(text)
                self.assertTrue(matched)

        false_cases = [
            "Read the credit card offer",
            "Scan my ID card",
            "Identify the business card",
            "Read the gift card balance",
        ]
        for text in false_cases:
            with self.subTest(text=text):
                matched, _reason = model_router.moondream_provider.card_mode_reason(text)
                self.assertFalse(matched)

    def test_credit_card_goal_does_not_use_playing_card_prompt(self):
        prompt, short_goal, _original_chars, source = model_router.moondream_provider.adapt_task_prompt(
            [{"role": "user", "content": "Read the credit card offer in this mail"}],
            {
                "capability": "structured_visual_understanding",
                "stage_goal": "Read the credit card offer in this mail",
            },
        )

        self.assertEqual(source, "stage_goal")
        self.assertEqual(short_goal, "Read the credit card offer in this mail")
        self.assertNotEqual(prompt, model_router.moondream_provider.CARD_IDENTIFICATION_PROMPT)

    def test_prompt_uses_first_complete_stage_goal_sentence(self):
        prompt, short_task, _chars, source = model_router.moondream_provider.adapt_task_prompt(
            [{"role": "user", "content": "Fallback provider prompt."}],
            {
                "stage_goal": "Identify every visible package. Then describe routing metadata.",
                "goal": "Different user goal",
                "capability": "structured_visual_understanding",
            },
        )
        self.assertEqual(source, "stage_goal")
        self.assertEqual(short_task, "Identify every visible package.")
        self.assertEqual(
            prompt,
            "Answer the visual task directly and briefly: Identify every visible package.",
        )

    def test_card_response_is_normalized_to_rank_and_suit(self):
        original = (
            "The card is a Queen of Spades, depicted in black ink on a white background."
        )
        create = Mock(return_value=response(
            original
        ))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(model_router, "_create_moondream_client", return_value=client), \
             patch.dict(model_router.os.environ, {
                 "MOONDREAM_API_KEY": "test-key", "ENABLE_MOONDREAM_CLOUD": "true",
             }), self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router._moondream_cloud_executor(
                self.profile(), [{"role": "user", "content": "Identify the cards and their properties"}],
                [self.image_bytes()],
                {
                    "capability": "structured_visual_understanding",
                    "goal": "Identify the cards and their properties",
                },
            )

        self.assertEqual(model_router._response_text(result.response), "Queen of spades.")
        output = "\n".join(logs.output)
        self.assertIn("card_mode=true", output)
        self.assertIn("matched identify+cards", output)
        self.assertIn(
            'original_task_goal="Identify the cards and their properties"',
            output,
        )
        self.assertIn("postprocess=card_rank_suit_only", output)
        self.assertIn("[Moondream] final_prompt_sent=", output)
        self.assertIn("[Moondream] raw_response=", output)
        self.assertIn("card_prompt_variant=multi_card_left_to_right", output)
        self.assertIn('[Moondream] extracted_cards=["Queen of spades"]', output)
        self.assertIn('[Moondream] deduped_cards=["Queen of spades"]', output)
        self.assertIn("mirrored_corner_filter_applied=false", output)
        self.assertIn('[Moondream] final_card_response="Queen of spades."', output)

    def test_multiple_card_response_extracts_all_rank_suit_pairs(self):
        text = "I can see a Jack of Hearts and a King of Spades."
        self.assertEqual(
            model_router.moondream_provider.normalize_card_response(text),
            "Jack of hearts, King of spades.",
        )

    def test_mirrored_corner_hallucination_prefers_first_card(self):
        text = "The card appears to be a six of hearts and a nine of spades."
        details = model_router.moondream_provider.card_response_details(text)
        self.assertEqual(details["final_response"], "Six of hearts.")
        self.assertTrue(details["mirrored_corner_filter_applied"])

    def test_lower_right_wording_filters_second_card(self):
        text = (
            "Six of hearts; the lower-right upside-down corner looks like "
            "nine of spades."
        )
        details = model_router.moondream_provider.card_response_details(text)
        self.assertEqual(details["final_response"], "Six of hearts.")
        self.assertTrue(details["mirrored_corner_filter_applied"])

    def test_card_rank_abbreviations_and_duplicates_are_normalized(self):
        text = "Q of spades, Queen of Spade, A of hearts, ten of clubs."
        details = model_router.moondream_provider.card_response_details(text)
        self.assertEqual(
            details["final_response"],
            "Queen of spades, Ace of hearts, Ten of clubs.",
        )
        self.assertEqual(len(details["extracted_cards"]), 4)
        self.assertEqual(len(details["deduped_cards"]), 3)

    def test_compact_card_notation_is_normalized(self):
        cases = {
            "J♦": "Jack of diamonds.",
            "J♠": "Jack of spades.",
            "Q♥": "Queen of hearts.",
            "10♣": "Ten of clubs.",
            "JD": "Jack of diamonds.",
            "J D": "Jack of diamonds.",
            "J-diamond": "Jack of diamonds.",
            "J diamond": "Jack of diamonds.",
            "JS": "Jack of spades.",
            "J heart": "Jack of hearts.",
            "JC": "Jack of clubs.",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                details = model_router.moondream_provider.card_response_details(raw)
                self.assertEqual(details["final_response"], expected)
                self.assertTrue(details["compact_notation_detected"])
                self.assertTrue(details["card_result_valid"])

    def test_invalid_compact_card_notation_is_uncertain(self):
        for raw in ("J+", "It looks as though the card is obscured."):
            with self.subTest(raw=raw):
                details = model_router.moondream_provider.card_response_details(raw)
                self.assertEqual(details["final_response"], "Uncertain.")
                self.assertFalse(details["compact_notation_detected"])
                self.assertFalse(details["card_result_valid"])

    def test_verbose_and_standard_card_responses_are_normalized(self):
        cases = {
            "The only card in your hand is the jack of diamonds.": "Jack of diamonds.",
            "Ace of Clubs": "Ace of clubs.",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(
                    model_router.moondream_provider.normalize_card_response(raw),
                    expected,
                )

    def test_messy_card_responses_are_not_clean_or_valid(self):
        cases = {
            "King of Hearts Clubs Diamonds": ["Hearts", "Clubs", "Diamonds"],
            "King hearts clubs diamonds": ["hearts", "clubs", "diamonds"],
            "J diamonds spades": ["diamonds", "spades"],
            "The card may be hearts or clubs": ["hearts", "clubs"],
            "Uncertain": [],
        }
        for raw, expected_suits in cases.items():
            with self.subTest(raw=raw):
                details = model_router.moondream_provider.card_response_details(raw)
                self.assertFalse(details["card_result_valid"])
                self.assertFalse(details["card_result_clean"])
                self.assertEqual(details["conflicting_suits"], expected_suits)

    def test_clean_single_and_multiple_card_responses_are_high_confidence(self):
        for raw in (
            "J♦",
            "Jack of diamonds",
            "Ten of diamonds, Four of diamonds",
            "Ace of clubs.",
        ):
            with self.subTest(raw=raw):
                details = model_router.moondream_provider.card_response_details(raw)
                self.assertTrue(details["card_result_valid"])
                self.assertTrue(details["card_result_clean"])
                self.assertEqual(details["discarded_card_tokens"], [])

    def test_card_response_without_rank_suit_is_uncertain(self):
        self.assertEqual(
            model_router.moondream_provider.normalize_card_response(
                "I can see decorative playing cards but cannot read them."
            ),
            "Uncertain.",
        )

    def test_valid_moondream_card_skips_evaluator_and_fallback(self):
        completion = response("J♦")
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=completion))
            )
        )
        evaluator_messages = []

        def fake_executor(profile, messages, _images, _metadata):
            if profile.name == "judge":
                evaluator_messages.extend(messages)
                return model_router.ImplementationResult(response("YES"))
            return model_router.ImplementationResult(response("fallback"))

        policies = {"structured_visual_understanding": {
            "candidates": ["moondream", "fallback"], "evaluator": "judge",
            "cascade": "test",
        }}
        profiles = {
            "moondream": self.profile(),
            "fallback": model_router.ImplementationProfile("fallback", "fake"),
            "judge": model_router.ImplementationProfile("judge", "fake"),
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.object(model_router, "_create_moondream_client", return_value=client), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": fake_executor}), \
             patch.dict(model_router.os.environ, {
                 "MOONDREAM_API_KEY": "test-key", "ENABLE_MOONDREAM_CLOUD": "true",
             }), self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(
                capability="structured_visual_understanding",
                goal="Identify the cards and their properties",
                images=[self.image_bytes()],
            )

        self.assertEqual(result["response"], "Jack of diamonds.")
        self.assertEqual(result["implementation"], "moondream")
        self.assertEqual(evaluator_messages, [])
        output = "\n".join(logs.output)
        self.assertIn("compact_notation_detected=true", output)
        self.assertIn("card_result_valid=true", output)
        self.assertIn("skipped=true", output)
        self.assertIn("[Evaluator] skipped: single visible card for visual_representation_understanding", output)

    def test_gemini_fallback_card_response_uses_shared_normalizer(self):
        calls = []

        def fake_executor(profile, _messages, _images, _metadata):
            calls.append(profile.name)
            if profile.name == "moondream":
                return model_router.ImplementationResult(response("J+"))
            if profile.name == "judge":
                return model_router.ImplementationResult(response("NO"))
            return model_router.ImplementationResult(response(
                "The only card in your hand is the jack of diamonds."
            ))

        policies = {"structured_visual_understanding": {
            "candidates": ["moondream", "gemini"], "evaluator": "judge",
            "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("moondream", "gemini", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": fake_executor}):
            result = model_router.copilot_llm_call(
                capability="structured_visual_understanding",
                goal="Identify only the playing cards by rank and suit.",
            )

        self.assertEqual(calls, ["moondream", "judge", "gemini"])
        self.assertEqual(result["implementation"], "gemini")
        self.assertEqual(result["response"], "Jack of diamonds.")

    def test_unclean_moondream_card_result_keeps_evaluator_behavior(self):
        calls = []

        def fake_executor(profile, _messages, _images, _metadata):
            calls.append(profile.name)
            if profile.name == "moondream":
                details = model_router.moondream_provider.card_response_details(
                    "King of Hearts Clubs Diamonds"
                )
                return model_router.ImplementationResult(
                    response(details["final_response"]), details
                )
            if profile.name == "judge":
                return model_router.ImplementationResult(response("NO"))
            return model_router.ImplementationResult(response("Jack of diamonds"))

        policies = {"structured_visual_understanding": {
            "candidates": ["moondream", "gemini"], "evaluator": "judge",
            "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("moondream", "gemini", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": fake_executor}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(
                capability="structured_visual_understanding",
                goal="Identify only the playing cards by rank and suit.",
            )

        self.assertEqual(calls, ["moondream", "judge", "gemini"])
        self.assertEqual(result["response"], "Jack of diamonds.")
        output = "\n".join(logs.output)
        self.assertIn('conflicting_suits=["Hearts", "Clubs", "Diamonds"]', output)
        self.assertIn("card_result_valid=false", output)
        self.assertIn("card_result_clean=false", output)
        self.assertIn("skipped=false", output)
        self.assertIn("reason=no clear single-card identification", output)

    def test_single_card_evaluator_skip_helper_examples(self):
        should_skip = model_router.should_skip_evaluator_for_single_card
        self.assertTrue(should_skip(
            "ten of diamonds", "anything", "visual_representation_understanding"
        ))
        self.assertTrue(should_skip(
            "The card is the ten of diamonds.", "anything", "visual_representation_understanding"
        ))
        self.assertTrue(should_skip(
            '{"cards": [{"rank": "10", "suit": "diamonds"}]}',
            "anything",
            "visual_representation_understanding",
        ))
        self.assertTrue(should_skip(
            '{"card_count": 1, "rank": "10", "suit": "diamonds"}',
            "anything",
            "visual_representation_understanding",
        ))
        self.assertTrue(should_skip(
            "The card is the ten of diamonds.", "playing_card_reader", "general_reasoning"
        ))
        self.assertFalse(should_skip(
            "The card is the ten of diamonds.", "anything", "general_reasoning"
        ))
        self.assertFalse(should_skip(
            "I see multiple cards.", "anything", "visual_representation_understanding"
        ))
        self.assertFalse(should_skip(
            "There are two cards.", "anything", "visual_representation_understanding"
        ))
        self.assertFalse(should_skip(
            "unclear", "anything", "visual_representation_understanding"
        ))
        self.assertFalse(should_skip(
            "I cannot tell", "anything", "visual_representation_understanding"
        ))
        self.assertFalse(should_skip(
            "possibly ten of diamonds", "anything", "visual_representation_understanding"
        ))
        self.assertFalse(should_skip(
            "ten of diamonds, jack of clubs", "anything", "visual_representation_understanding"
        ))

    def test_non_moondream_model_receives_original_messages(self):
        messages = [{"role": "user", "content": "Keep this complete original prompt."}]
        profile = model_router.ImplementationProfile("gemini", "model", model="gemini/test")
        with patch.object(model_router, "call_model", return_value=response("ok")) as call_model:
            model_router._model_executor(profile, messages, [b"image"], {})
        self.assertEqual(call_model.call_args.args[1], messages)

    def test_extracts_text_content_part(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=[{"type": "text", "text": "Visible answer."}]
        ))])
        self.assertEqual(model_router._moondream_response_text(completion), "Visible answer.")

    def test_prepares_opencv_bgr_ndarray_as_rgb_jpeg(self):
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # OpenCV BGR blue.

        payload, media_type, dimensions = model_router._prepare_moondream_image(frame)

        self.assertEqual(media_type, "image/jpeg")
        self.assertEqual(dimensions, (40, 20))
        with Image.open(io.BytesIO(payload)) as decoded:
            red, _green, blue = decoded.convert("RGB").getpixel((20, 10))
        self.assertGreater(blue, red)

    def test_executor_accepts_streaming_ndarray_frame(self):
        create = Mock(return_value=response("The frame is visible."))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        with patch.object(model_router, "_create_moondream_client", return_value=client), \
             patch.dict(model_router.os.environ, {
                 "MOONDREAM_API_KEY": "test-key", "ENABLE_MOONDREAM_CLOUD": "true",
             }):
            result = model_router._moondream_cloud_executor(
                self.profile(), [{"role": "user", "content": "Describe it"}], [frame], {}
            )

        self.assertEqual(model_router._response_text(result.response), "The frame is visible.")
        image_url = create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))

    def test_pil_and_data_uri_inputs_are_safely_reencoded_for_request(self):
        jpeg = self.image_bytes()
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        inputs = [Image.new("RGBA", (16, 12), "white"), data_uri]
        for source in inputs:
            with self.subTest(source_type=type(source).__name__):
                create = Mock(return_value=response("Visible."))
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                with patch.object(model_router, "_create_moondream_client", return_value=client), \
                     patch.dict(model_router.os.environ, {
                         "MOONDREAM_API_KEY": "test-key", "ENABLE_MOONDREAM_CLOUD": "true",
                     }):
                    model_router._moondream_cloud_executor(
                        self.profile(), [{"role": "user", "content": "Describe it"}],
                        [source], {},
                    )
                part = create.call_args.kwargs["messages"][0]["content"][1]
                self.assertEqual(part["type"], "image_url")
                self.assertTrue(part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_empty_or_malformed_response_is_soft_failure(self):
        create = Mock(return_value=SimpleNamespace(choices=[]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(model_router, "_create_moondream_client", return_value=client), \
             patch.dict(model_router.os.environ, {
                 "MOONDREAM_API_KEY": "test-key", "ENABLE_MOONDREAM_CLOUD": "true",
             }), self.assertLogs(model_router.logger, level="INFO") as logs:
            with self.assertRaisesRegex(RuntimeError, "empty or malformed"):
                model_router._moondream_cloud_executor(
                    self.profile(), [{"role": "user", "content": "Describe it"}],
                    [self.image_bytes()], {},
                )
        self.assertIn("continuing cascade", "\n".join(logs.output))

    def test_500_starts_cooldown_and_next_call_skips_network(self):
        error = RuntimeError("FAL backend error: 500")
        error.status_code = 500
        error.request_id = "request-123"
        create = Mock(side_effect=error)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        environment = {
            "MOONDREAM_API_KEY": "test-key", "ENABLE_MOONDREAM_CLOUD": "true",
            "MOONDREAM_FAILURE_COOLDOWN_SECONDS": "60",
        }
        with patch.object(model_router, "_create_moondream_client", return_value=client), \
             patch.dict(model_router.os.environ, environment), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            with self.assertRaisesRegex(RuntimeError, "FAL backend"):
                model_router._moondream_cloud_executor(
                    self.profile(), [{"role": "user", "content": "Describe it"}],
                    [self.image_bytes()], {"capability": "general_reasoning", "goal": "Describe it"},
                )
            with self.assertRaisesRegex(RuntimeError, "cooldown"):
                model_router._moondream_cloud_executor(
                    self.profile(), [{"role": "user", "content": "Describe it"}],
                    [self.image_bytes()], {},
                )

        self.assertEqual(create.call_count, 1)
        output = "\n".join(logs.output)
        self.assertIn("failed status=500 request_id=request-123", output)
        self.assertIn("temporarily disabled reason=provider_backend_failure cooldown_seconds=60", output)

    def test_timeout_failure_continues_policy_cascade_to_gemini(self):
        create = Mock(side_effect=TimeoutError("timed out"))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        calls = []

        def fake_executor(profile, _messages, _images, _metadata):
            calls.append(profile.name)
            return model_router.ImplementationResult(response("Gemini recovered."))

        policies = {"general_reasoning": {
            "candidates": ["moondream", "gemini"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {
            "moondream": self.profile(),
            "gemini": model_router.ImplementationProfile("gemini", "fake"),
            "judge": model_router.ImplementationProfile("judge", "fake"),
        }
        with patch.object(model_router, "_create_moondream_client", return_value=client), \
             patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": fake_executor}), \
             patch.dict(model_router.os.environ, {
                 "MOONDREAM_API_KEY": "test-key", "ENABLE_MOONDREAM_CLOUD": "true",
             }):
            result = model_router.copilot_llm_call(
                capability="general_reasoning", images=[self.image_bytes()]
            )

        self.assertEqual(calls, ["gemini"])
        self.assertEqual(result["response"], "Gemini recovered.")
        self.assertEqual(result["implementation"], "gemini")


class TestStreamingLatencyControls(unittest.TestCase):
    def test_streaming_vlm_applies_original_jpeg_exif_before_resize(self):
        source = Image.new("RGB", (80, 40), "white")
        exif = Image.Exif()
        exif[274] = 6
        encoded = io.BytesIO()
        source.save(encoded, format="JPEG", exif=exif)
        sideways_decoded_frame = np.zeros((40, 80, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            model_router.os.environ,
            {"MOONDREAM_DEBUG_IMAGE_PATH": str(Path(temp_dir) / "corrected.jpg")},
        ), patch.object(model_router, "_STREAMING_VLM_DEBUG_SAVED", False):
            processed, _elapsed_ms = model_router._preprocess_streaming_images(
                [sideways_decoded_frame],
                "structured_visual_understanding",
                original_image=encoded.getvalue(),
            )
            debug_path = Path(temp_dir) / "corrected.jpg"
            self.assertTrue(debug_path.exists())
            with Image.open(io.BytesIO(processed[0])) as image:
                self.assertEqual(image.size, (40, 80))
            with Image.open(debug_path) as image:
                self.assertEqual(image.size, (40, 80))

    def test_streaming_vlm_image_is_downscaled_once(self):
        frame = np.zeros((1200, 1800, 3), dtype=np.uint8)
        with patch.dict(model_router.os.environ, {
            "STREAMING_VLM_MAX_DIMENSION": "640",
            "STREAMING_VLM_JPEG_QUALITY": "70",
        }), self.assertLogs(model_router.logger, level="INFO") as logs:
            processed, elapsed_ms = model_router._preprocess_streaming_images(
                [frame], "general_reasoning"
            )

        with Image.open(io.BytesIO(processed[0])) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(max(image.size), 640)
        self.assertGreaterEqual(elapsed_ms, 0)
        output = "\n".join(logs.output)
        self.assertIn("original_dimensions=1800x1200", output)
        self.assertIn("max_dimension=640", output)

    def test_streaming_ocr_uses_separate_resolution(self):
        frame = np.zeros((1200, 1800, 3), dtype=np.uint8)
        with patch.dict(model_router.os.environ, {
            "OCR_MAX_DIMENSION": "1024", "OCR_JPEG_QUALITY": "80",
        }):
            processed, _elapsed_ms = model_router._preprocess_streaming_images([frame], "ocr")
        with Image.open(io.BytesIO(processed[0])) as image:
            self.assertEqual(max(image.size), 1024)

    def test_valid_card_result_skips_evaluator_and_streaming_escalation(self):
        calls = []

        def executor(profile, _messages, _images, _metadata):
            calls.append(profile.name)
            if profile.name == "judge":
                return model_router.ImplementationResult(response("NO"))
            if profile.name == "moondream":
                return model_router.ImplementationResult(response("The visible card is the Ace of Spades."))
            self.fail("streaming guard should avoid the next candidate")

        policies = {"general_reasoning": {
            "candidates": ["moondream", "gemini"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("moondream", "gemini", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(
                capability="general_reasoning", goal="Identify the playing card rank and suit",
                metadata={"streaming": True, "streaming_execution_id": 7},
            )

        self.assertEqual(calls, ["moondream", "judge"])
        self.assertEqual(result["implementation"], "moondream")
        output = "\n".join(logs.output)
        self.assertIn("[Evaluator] decision=YES source=streaming_acceptance_guard", output)
        self.assertIn("[Streaming Timing] execution=7", output)
        self.assertIn("selected=moondream", output)

    def test_uncertain_card_answer_is_not_guarded(self):
        self.assertFalse(model_router._streaming_acceptance_guard(
            "Identify the playing card", "I am not sure, but it may be the Ace of Spades."
        ))

    def test_stop_check_cancels_before_evaluator_and_next_candidate(self):
        stopped = {"value": False}
        calls = []

        def executor(profile, _messages, _images, _metadata):
            calls.append(profile.name)
            stopped["value"] = True
            return model_router.ImplementationResult(response("The Ace of Hearts."))

        policies = {"general_reasoning": {
            "candidates": ["first", "second"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("first", "second", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            with self.assertRaises(model_router.StreamingExecutionCancelled):
                model_router.copilot_llm_call(
                    capability="general_reasoning",
                    metadata={"streaming": True, "_streaming_cancelled": lambda: stopped["value"]},
                )
        self.assertEqual(calls, ["first"])

    def test_evaluator_debug_mode_parses_decision_and_logs_reason(self):
        calls = []

        def executor(profile, messages, _images, _metadata):
            calls.append(profile.name)
            if profile.name == "judge":
                self.assertIn("DECISION: YES or NO", messages[0]["content"])
                return model_router.ImplementationResult(response(
                    "DECISION: NO\nREASON: The answer is too vague."
                ))
            return model_router.ImplementationResult(response(f"response from {profile.name}"))

        policies = {"general_reasoning": {
            "candidates": ["first", "second"], "evaluator": "judge", "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("first", "second", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             patch.dict(model_router.os.environ, {"EVALUATOR_DEBUG_REASON": "true"}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(capability="general_reasoning")

        self.assertEqual(calls, ["first", "judge", "second"])
        self.assertEqual(result["implementation"], "second")
        self.assertIn('[Evaluator] reason="The answer is too vague."', "\n".join(logs.output))


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
        self.assertEqual(
            policies["ocr"]["candidates"],
            ["mistral_ocr", "google_vision"],
        )
        self.assertEqual(policies["ocr"]["evaluator"], "gpt4o-mini")
        self.assertEqual(policies["ocr"]["cascade"], "ocr")
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
        self.assertFalse(config["routing_enabled"])
        self.assertEqual(config["system_model"], "llama_planner")
        self.assertEqual(config["default_llm_when_routing_disabled"], "moondream_cloud")

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

    def test_reasoning_cascade_uses_moondream_then_gemini_then_gpt4o(self):
        policy = model_router.load_execution_policies()["general_reasoning"]
        self.assertEqual(
            policy["candidates"],
            ["moondream_cloud", "gemini_flash_lite", "gpt4o"],
        )
        self.assertEqual(policy["evaluator"], "gpt4o-mini")

class TestObjectDetectorRouting(unittest.TestCase):
    def setUp(self):
        self.profile = model_router.ImplementationProfile(
            "yolo", "yolo", model_name="configured-default.pt"
        )

    def assert_detector(self, labels, expected_model, expected_log):
        normalized = model_router._target_labels({"target_labels": labels})
        with self.assertLogs(model_router.logger, level="INFO") as logs:
            model_name, _ = model_router._object_detector_model(self.profile, normalized)
        self.assertEqual(model_name, expected_model)
        self.assertIn(expected_log, "\n".join(logs.output))

    def test_coco_targets_route_to_yolo11(self):
        for label in ("car", "chair"):
            with self.subTest(label=label):
                self.assert_detector(
                    [label], model_router.YOLO11_MODEL,
                    "detector=yolo11 reason=all_targets_in_coco",
                )

    def test_non_coco_targets_route_to_yoloworld(self):
        for label in ("exit sign", "soy sauce bottle"):
            with self.subTest(label=label):
                self.assert_detector(
                    [label], model_router.YOLOWORLD_MODEL,
                    "detector=yoloworld reason=non_coco_targets",
                )

    def test_target_normalization_happens_before_routing(self):
        self.assert_detector(
            ["  CAR  "], model_router.YOLO11_MODEL,
            "target_labels=['car']",
        )

    def test_missing_or_empty_targets_keep_configured_default(self):
        for metadata in ({}, {"target_labels": []}, {"target_labels": ["  "]}):
            with self.subTest(metadata=metadata):
                normalized = model_router._target_labels(metadata)
                with self.assertLogs(model_router.logger, level="INFO") as logs:
                    model_name, detector = model_router._object_detector_model(
                        self.profile, normalized
                    )
                self.assertEqual(model_name, "configured-default.pt")
                self.assertEqual(detector, "default")
                self.assertIn(
                    "detector=default reason=no_target_labels", "\n".join(logs.output)
                )


class TestOcrCascadeRouting(unittest.TestCase):
    @staticmethod
    def _image_bytes():
        image_bytes = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
        return image_bytes.getvalue()

    def _run_ocr(self, executor):
        policies = {"ocr": {
            "candidates": ["mistral_ocr", "google_vision"],
            "evaluator": "gpt4o-mini",
            "cascade": "ocr",
            "specialized": True,
        }}
        profiles = {
            "mistral_ocr": model_router.ImplementationProfile("mistral_ocr", "fake"),
            "google_vision": model_router.ImplementationProfile("google_vision", "fake"),
            "gpt4o-mini": model_router.ImplementationProfile("gpt4o-mini", "fake"),
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router.copilot_llm_call(capability="ocr", images=[b"image"])
        return result, "\n".join(logs.output)

    def test_mistral_is_tried_and_accepted_first(self):
        calls = []

        def executor(profile, *_args):
            calls.append(profile.name)
            if profile.name == "gpt4o-mini":
                return model_router.ImplementationResult(response("YES"))
            return model_router.ImplementationResult(
                response("A sufficiently long receipt"),
                {"text": "A sufficiently long receipt", "accepted": True},
            )

        result, logs = self._run_ocr(executor)

        self.assertEqual(calls, ["mistral_ocr", "gpt4o-mini"])
        self.assertEqual(result["implementation"], "mistral_ocr")
        self.assertIn("[OCR] trying=mistral_ocr", logs)
        self.assertIn("[OCR] implementation=mistral_ocr latency_ms=", logs)
        self.assertIn(
            "[Execution Policy] capability=ocr candidate=mistral_ocr", logs
        )
        self.assertIn(
            "[Execution Policy] capability=ocr evaluator=gpt4o-mini decision=YES", logs
        )

    def test_insufficient_mistral_falls_back_to_google_vision(self):
        calls = []

        def executor(profile, *_args):
            calls.append(profile.name)
            if profile.name == "gpt4o-mini":
                return model_router.ImplementationResult(response("NO"))
            if profile.name == "mistral_ocr":
                return model_router.ImplementationResult(
                    response("Partial vague text"),
                    {"text": "Partial vague text", "accepted": True},
                )
            return model_router.ImplementationResult(
                response("Accurate Google Vision text"),
                {"text": "Accurate Google Vision text", "accepted": True},
            )

        result, logs = self._run_ocr(executor)

        self.assertEqual(
            calls, ["mistral_ocr", "gpt4o-mini", "google_vision"]
        )
        self.assertEqual(result["implementation"], "google_vision")
        self.assertIn(
            "[Execution Policy] capability=ocr evaluator=gpt4o-mini decision=NO", logs
        )
        self.assertIn("[OCR] trying=google_vision", logs)
        self.assertIn("[OCR] implementation=google_vision latency_ms=", logs)

    def test_empty_mistral_output_falls_back_without_evaluation(self):
        calls = []

        def executor(profile, *_args):
            calls.append(profile.name)
            if profile.name == "mistral_ocr":
                return model_router.ImplementationResult(response(""), {"text": ""})
            if profile.name == "gpt4o-mini":
                self.fail("empty Mistral output should not call the evaluator")
            return model_router.ImplementationResult(response("Google Vision text"))

        result, _logs = self._run_ocr(executor)

        self.assertEqual(calls, ["mistral_ocr", "google_vision"])
        self.assertEqual(result["implementation"], "google_vision")

    def test_failed_mistral_output_falls_back_to_google_vision(self):
        calls = []

        def executor(profile, *_args):
            calls.append(profile.name)
            if profile.name == "mistral_ocr":
                raise RuntimeError("mistral_ocr_failed")
            if profile.name == "gpt4o-mini":
                self.fail("failed Mistral output should not call the evaluator")
            return model_router.ImplementationResult(response("Google Vision text"))

        result, _logs = self._run_ocr(executor)

        self.assertEqual(calls, ["mistral_ocr", "google_vision"])
        self.assertEqual(result["implementation"], "google_vision")

    def test_mistral_executor_sends_jpeg_data_uri_and_joins_pages(self):
        captured = {}

        class Ocr:
            def process(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(pages=[
                    SimpleNamespace(markdown="First page"),
                    SimpleNamespace(markdown=""),
                    SimpleNamespace(markdown="Second page"),
                ])

        class Mistral:
            def __init__(self):
                captured["client_created"] = True
                self.ocr = Ocr()

        mistralai_module = ModuleType("mistralai")
        mistralai_module.__path__ = []
        mistralai_client_module = ModuleType("mistralai.client")
        mistralai_client_module.Mistral = Mistral
        profile = model_router.ImplementationProfile("mistral_ocr", "mistral_ocr")
        with patch.dict(sys.modules, {
                 "mistralai": mistralai_module,
                 "mistralai.client": mistralai_client_module,
             }), \
             patch.dict(model_router.os.environ, {"MISTRAL_OCR_MODEL": "mistral-ocr-latest"}), \
             self.assertLogs(model_router.logger, level="INFO") as logs:
            result = model_router._mistral_ocr_executor(
                profile, [], [self._image_bytes()], {}
            )

        self.assertTrue(captured["client_created"])
        self.assertEqual(captured["model"], "mistral-ocr-latest")
        document = captured["document"]
        self.assertEqual(document["type"], "image_url")
        self.assertTrue(document["image_url"].startswith("data:image/jpeg;base64,"))
        encoded = document["image_url"].split(",", 1)[1]
        self.assertTrue(model_router.base64.b64decode(encoded).startswith(b"\xff\xd8"))
        self.assertEqual(model_router._response_text(result.response), "First page\nSecond page")
        self.assertIn(
            "[Mistral OCR] model=mistral-ocr-latest input=image_data_uri payload_kb=",
            "\n".join(logs.output),
        )

    def test_mistral_import_failure_is_recoverable_and_logs_import_path(self):
        profile = model_router.ImplementationProfile("mistral_ocr", "mistral_ocr")
        with patch.dict(sys.modules, {"mistralai": None, "mistralai.client": None}), \
             self.assertLogs(model_router.logger, level="WARNING") as logs, \
             self.assertRaisesRegex(RuntimeError, "mistral_ocr_import_failed"):
            model_router._mistral_ocr_executor(
                profile, [], [self._image_bytes()], {}
            )

        self.assertIn(
            "import_failed path=mistralai.client.Mistral",
            "\n".join(logs.output),
        )

    def test_missing_tesseract_binary_raises_recoverable_provider_error(self):
        class TesseractNotFoundError(Exception):
            pass

        fake_pytesseract = SimpleNamespace(
            Output=SimpleNamespace(DICT="dict"),
            TesseractNotFoundError=TesseractNotFoundError,
            image_to_data=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TesseractNotFoundError("binary missing")
            ),
        )
        profile = model_router.ImplementationProfile("tesseract_local", "tesseract")

        with patch.dict(sys.modules, {"pytesseract": fake_pytesseract}), \
             self.assertLogs(model_router.logger, level="WARNING") as logs, \
             self.assertRaisesRegex(RuntimeError, "tesseract_binary_unavailable"):
            model_router._tesseract_executor(
                profile, [], [self._image_bytes()], {}
            )

        self.assertIn(
            "tesseract insufficient reason=tesseract_binary_unavailable",
            "\n".join(logs.output),
        )

    def test_empty_tesseract_text_fails_but_short_text_reaches_evaluator(self):
        profile = model_router.ImplementationProfile("tesseract_local", "tesseract")
        fake_pytesseract = SimpleNamespace(
            Output=SimpleNamespace(DICT="dict"),
            TesseractNotFoundError=type("TesseractNotFoundError", (Exception,), {}),
            image_to_data=lambda *_args, **_kwargs: {"text": [], "conf": []},
        )
        with patch.dict(sys.modules, {"pytesseract": fake_pytesseract}), \
             self.assertRaisesRegex(RuntimeError, "no_text"):
            model_router._tesseract_executor(profile, [], [self._image_bytes()], {})

        fake_pytesseract.image_to_data = lambda *_args, **_kwargs: {
            "text": ["tiny"],
            "conf": [20],
        }
        with patch.dict(sys.modules, {"pytesseract": fake_pytesseract}):
            result = model_router._tesseract_executor(
                profile, [], [self._image_bytes()], {}
            )
        self.assertEqual(model_router._response_text(result.response), "tiny")


class TestAtomicCopilotCall(unittest.TestCase):
    def test_generation_prompt_requires_single_line_plain_audio_text(self):
        calls = []

        def executor(_profile, messages, _images, _metadata):
            calls.append(messages)
            return model_router.ImplementationResult(response(
                "From left to right:\n1. **Jack of Spades**\n2. `Ten of Spades`"
            ))

        policies = {"general_reasoning": {
            "candidates": ["vlm"], "evaluator": None, "cascade": None,
        }}
        profiles = {"vlm": model_router.ImplementationProfile("vlm", "fake")}
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(
                capability="general_reasoning", goal="Identify the cards"
            )

        prompt = calls[0][0]["content"]
        self.assertIn("audio-friendly plain text in one line", prompt)
        self.assertIn("Do not use Markdown", prompt)
        self.assertIn("commas or semicolons", prompt)
        self.assertEqual(
            result["response"],
            "Jack of spades, Ten of spades.",
        )
        self.assertNotIn("\n", result["response"])
        self.assertNotIn("**", result["response"])

    def test_evaluator_prompt_does_not_receive_generation_format_prompt(self):
        calls = []

        def executor(profile, messages, images, metadata):
            calls.append((profile.name, messages, images, metadata))
            text = "YES" if profile.name == "judge" else "Useful answer."
            return model_router.ImplementationResult(response(text))

        policies = {"general_reasoning": {
            "candidates": ["candidate", "fallback"],
            "evaluator": "judge",
            "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("candidate", "fallback", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            model_router.copilot_llm_call(capability="general_reasoning", goal="Describe it")

        evaluator_prompt = calls[1][1][0]["content"]
        self.assertNotIn(model_router.AUDIO_RESPONSE_PROMPT, evaluator_prompt)
        self.assertIn("Output exactly one token: YES or NO", evaluator_prompt)

    def test_streaming_single_stage_uses_concise_audio_prompt(self):
        profiles = {
            "default": model_router.ImplementationProfile(
                "default", "model", model="test/default"
            ),
            "system": model_router.ImplementationProfile(
                "system", "model", model="test/system"
            ),
        }
        config = {
            "planner_enabled": False,
            "routing_enabled": False,
            "system_model": "system",
            "default_llm_when_routing_disabled": "default",
        }
        with patch.object(model_router, "load_global_execution_config", return_value=config), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.object(model_router, "call_model", return_value=response("Brief answer")) as call_model:
            model_router.single_stage_llm_call(
                task="Categorize this mail",
                images=[b"image"],
                metadata={"streaming": True},
            )

        messages = call_model.call_args.args[1]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("1-2 short, audio-friendly sentences", messages[0]["content"])
        self.assertIn("likely category and one brief reason", messages[0]["content"])

    def test_non_streaming_single_stage_keeps_detailed_prompt_available(self):
        profiles = {
            "default": model_router.ImplementationProfile(
                "default", "model", model="test/default"
            ),
            "system": model_router.ImplementationProfile(
                "system", "model", model="test/system"
            ),
        }
        config = {
            "planner_enabled": False,
            "routing_enabled": False,
            "system_model": "system",
            "default_llm_when_routing_disabled": "default",
        }
        with patch.object(model_router, "load_global_execution_config", return_value=config), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.object(model_router, "call_model", return_value=response("Detailed answer")) as call_model:
            model_router.single_stage_llm_call(
                task="Describe this document",
                images=[b"image"],
                metadata={"streaming": False},
            )

        messages = call_model.call_args.args[1]
        self.assertFalse(any(
            model_router.STREAMING_RESPONSE_PROMPT in message["content"]
            for message in messages
        ))

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

    def test_single_candidate_does_not_call_configured_evaluator(self):
        calls = []

        def executor(profile, *_args):
            calls.append(profile.name)
            if profile.name == "judge":
                self.fail("single-candidate capability should not call evaluator")
            return model_router.ImplementationResult(response("Direct result"))

        policies = {"ocr": {
            "candidates": ["reader"],
            "evaluator": "judge",
            "cascade": "test",
        }}
        profiles = {
            "reader": model_router.ImplementationProfile("reader", "fake"),
            "judge": model_router.ImplementationProfile("judge", "fake"),
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}):
            result = model_router.copilot_llm_call(capability="ocr")

        self.assertEqual(calls, ["reader"])
        self.assertEqual(result["implementation"], "reader")

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
                metadata={
                    "previous_stage_artifact": {
                        "text": "The exit is on the right.",
                        "detections": [{"label": "exit"}],
                    },
                    "image_url": "data:image/jpeg;base64,secret",
                    "image_base64": "secret",
                    "image": b"secret",
                },
                images=[b"candidate-image"],
            )

        self.assertEqual([call[0] for call in calls], ["llava", "gpt4o"])
        self.assertEqual(result["response"], "Turn right toward the exit.")
        self.assertEqual(result["implementation"], "llava")
        self.assertIn("previous-stage artifact", calls[0][1][-2]["content"])
        evaluator_prompt = calls[1][1][0]["content"]
        self.assertIn("Original user request or tool goal: Guide me to the exit", evaluator_prompt)
        self.assertIn("Previous-stage textual outputs, if any: The exit is on the right.", evaluator_prompt)
        self.assertNotIn('"detections"', evaluator_prompt)
        self.assertIn("The blue envelope is junk", evaluator_prompt)
        self.assertIn("The bottom mailer is a junk credit card offer", evaluator_prompt)
        self.assertIn("promotional flyer for a dental office", evaluator_prompt)
        self.assertIn("Partial but useful answers should usually be accepted", evaluator_prompt)
        self.assertIn("uncertainty is clearly communicated", evaluator_prompt)
        self.assertIn("Output exactly one token: YES or NO", evaluator_prompt)
        self.assertEqual(calls[1][2], [])
        self.assertEqual(calls[1][3], {
            "temperature": 0,
            "max_tokens": 3,
            "capability": "navigation",
            "evaluator": True,
        })
        self.assertNotIn("image", calls[1][3])
        self.assertNotIn("image_url", calls[1][3])
        self.assertNotIn("image_base64", calls[1][3])

    def test_debug_reason_uses_separate_text_only_evaluator_call(self):
        calls = []

        def executor(profile, messages, images, metadata):
            calls.append((profile.name, messages, images, metadata))
            prompt = messages[0].get("content", "")
            if "Output exactly one token: YES or NO" in prompt:
                return model_router.ImplementationResult(response("YES"))
            if "Output only the concise reason sentence" in prompt:
                return model_router.ImplementationResult(response(
                    "Categorizes one item and clearly marks the other as unreadable."
                ))
            return model_router.ImplementationResult(response(
                "The flyer is junk mail; the top envelope is unreadable."
            ))

        policies = {"general_reasoning": {
            "candidates": ["candidate", "fallback"],
            "evaluator": "judge",
            "cascade": "test",
        }}
        profiles = {
            name: model_router.ImplementationProfile(name, "fake")
            for name in ("candidate", "fallback", "judge")
        }
        with patch.object(model_router, "load_execution_policies", return_value=policies), \
             patch.object(model_router, "load_implementation_profiles", return_value=profiles), \
             patch.dict(model_router.IMPLEMENTATION_EXECUTORS, {"fake": executor}), \
             self.assertLogs(model_router.logger, level="DEBUG") as logs:
            result = model_router.copilot_llm_call(
                capability="general_reasoning",
                goal="Categorize the mail",
                images=[b"candidate-image"],
            )

        self.assertEqual(result["implementation"], "candidate")
        self.assertEqual([call[0] for call in calls], ["candidate", "judge", "judge"])
        self.assertEqual(calls[1][2], [])
        self.assertEqual(calls[2][2], [])
        self.assertNotIn("image", calls[2][3])
        self.assertIn(
            '[Evaluator] decision=YES reason="Categorizes one item and clearly marks the other as unreadable."',
            "\n".join(logs.output),
        )

    def test_exit_target_is_inferred_and_evaluator_ignores_structured_grounding(self):
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
        self.assertNotIn('"label": "door"', prompt)
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
        evaluator_images = []

        def executor(profile, messages, images, _metadata):
            is_evaluation = "Output exactly one token: YES or NO" in messages[0].get("content", "")
            if profile.name == "gpt4o" and is_evaluation:
                evaluator_images.append(images)
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
        self.assertEqual(output.count("[Evaluator] image_included=false"), 2)
        self.assertEqual(output.count("[Evaluator] decision=NO"), 2)
        self.assertEqual(evaluator_images, [[], []])
        self.assertIn("selected implementation=gpt4o", output)
        self.assertEqual(result["response"], "response from gpt4o")

    def test_returns_best_response_when_later_candidate_fails(self):
        def executor(profile, messages, _images, _metadata):
            is_evaluation = "Output exactly one token: YES or NO" in messages[0].get("content", "")
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
            is_evaluation = "Output exactly one token: YES or NO" in messages[0].get("content", "")
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
        seen_messages = []

        def executor(profile, messages, _images, _metadata):
            calls.append(profile.name)
            seen_messages.extend(messages)
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
            result = model_router.copilot_llm_call(
                capability="navigation",
                messages=[{"role": "user", "content": "Guide me toward the selected seat."}],
            )

        self.assertEqual(calls, ["default"])
        self.assertTrue(any(
            message.get("content") == "Guide me toward the selected seat."
            for message in seen_messages
        ))
        self.assertEqual(result["implementation"], "default")
        output = "\n".join(logs.output)
        self.assertIn("planner_enabled=true routing_enabled=false", output)
        self.assertIn("system_model=system/test/system", output)
        self.assertIn("default_llm_when_routing_disabled=default/test/default", output)
        self.assertIn("capability=navigation selected implementation=default", output)

    def test_routing_disabled_uses_fixed_default_for_specialized_capability(self):
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

        self.assertEqual(calls, ["default"])
        self.assertEqual(result["implementation"], "default")


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
