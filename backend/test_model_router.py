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

    def test_policy_mode_preserves_exact_untruncated_tool_prompt(self):
        prompt = (
            "Identify important mail and briefly label each important item. "
            + "Keep this task-specific instruction. " * 200
            + "END_OF_TOOL_PROMPT"
        )
        create = Mock(return_value=response("Important: medical appointment letter."))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with patch.object(model_router, "_create_moondream_client", return_value=client), \
             patch.dict(model_router.os.environ, {
                 "MOONDREAM_API_KEY": "test-key",
                 "ENABLE_MOONDREAM_CLOUD": "true",
             }), self.assertLogs(model_router.logger, level="INFO") as logs:
            model_router._moondream_cloud_executor(
                self.profile(),
                [{"role": "user", "content": prompt}],
                [self.image_bytes()],
                {"preserve_original_prompt": True, "prompt_source": "tool_prompt"},
            )

        request_text = create.call_args.kwargs["messages"][0]["content"][0]["text"]
        self.assertEqual(request_text, prompt)
        output = "\n".join(logs.output)
        self.assertIn("prompt_adapter=none", output)
        self.assertIn("prompt_source=original_prompt_preserved", output)
        self.assertIn(f"prompt_chars_original={len(prompt)}", output)
        self.assertIn(f"prompt_chars_sent={len(prompt)}", output)

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

    def test_card_response_is_returned_without_postprocessing(self):
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

        self.assertEqual(model_router._response_text(result.response), original)
        output = "\n".join(logs.output)
        self.assertIn("card_mode=true", output)
        self.assertIn("matched identify+cards", output)
        self.assertIn(
            'original_task_goal="Identify the cards and their properties"',
            output,
        )
        self.assertIn("postprocess=none", output)
        self.assertIn("[Moondream] final_prompt_sent=", output)
        self.assertIn("[Moondream] raw_response=", output)
        self.assertIn("card_prompt_variant=multi_card_left_to_right", output)
        self.assertIn('[Moondream] extracted_cards=["Queen of spades"]', output)
        self.assertIn('[Moondream] deduped_cards=["Queen of spades"]', output)
        self.assertIn("mirrored_corner_filter_applied=false", output)
        self.assertIn('[Moondream] parsed_card_response="Queen of spades."', output)

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
        self.assertIn("empty or malformed response", "\n".join(logs.output))

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


class TestOcrProviderWrappers(unittest.TestCase):
    @staticmethod
    def _image_bytes():
        image_bytes = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
        return image_bytes.getvalue()

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

class TestSystemCall(unittest.TestCase):
    def test_system_llm_call_uses_yaml_global_implementation(self):
        global_config = {
            "system_model": "planner",
        }
        profiles = {
            "planner": model_router.ImplementationProfile("planner", "model", model="test/planner"),
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
