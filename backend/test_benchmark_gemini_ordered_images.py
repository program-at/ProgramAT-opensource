import unittest

from scripts.benchmark_gemini_ordered_images import benchmark_validation


class TestBenchmarkEvidenceValidation(unittest.TestCase):
    def test_remaining_card_mention_is_allowed(self):
        parsed = {
            "event_detected": True,
            "before_cards": ["nine of hearts", "jack of diamonds"],
            "after_cards": ["jack of diamonds"],
            "played_card": "nine of hearts",
            "confidence": 0.9,
            "evidence": "The nine of hearts was removed while the jack of diamonds remains.",
        }
        self.assertEqual(
            benchmark_validation(parsed),
            (True, "accepted_remaining_card_mentioned"),
        )

    def test_different_removed_card_is_rejected(self):
        parsed = {
            "event_detected": True,
            "before_cards": ["nine of hearts", "jack of diamonds"],
            "after_cards": ["jack of diamonds"],
            "played_card": "nine of hearts",
            "confidence": 0.9,
            "evidence": "The jack of diamonds was removed.",
        }
        passed, reason = benchmark_validation(parsed)
        self.assertFalse(passed)
        self.assertIn("jack of diamonds", reason)


if __name__ == "__main__":
    unittest.main()
