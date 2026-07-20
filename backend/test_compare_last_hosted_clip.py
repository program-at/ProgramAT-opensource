import unittest

from scripts.compare_last_hosted_clip import normalize_expected_card


class TestExpectedCardNormalization(unittest.TestCase):
    def test_singular_and_plural_suits_normalize_identically(self):
        self.assertEqual(
            normalize_expected_card("two of spade"),
            normalize_expected_card("two of spades"),
        )
        self.assertEqual(normalize_expected_card(" Two  of  Spade "), "two of spades")


if __name__ == "__main__":
    unittest.main()
