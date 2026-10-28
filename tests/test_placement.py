"""Characterization tests for the pure ad placement functions."""
import unittest

from src.ad_integration.core.placement import format_ad_block, place_ad

BLOCK = format_ad_block("Head", "Desc", "Buy [now]")


class PlacementTests(unittest.TestCase):
    def test_ad_block_format(self) -> None:
        self.assertEqual(BLOCK, "---\n\n# Head\n\n### Desc\n\n### [Buy [now](#)\n\n---")
        self.assertIn("### [Learn more](#)", format_ad_block("h", "d", "  "))

    def test_first_and_last(self) -> None:
        self.assertEqual(place_ad("Body text.", BLOCK, "first"), f"{BLOCK}\n\nBody text.")
        self.assertEqual(place_ad("Body text.", BLOCK, "last"), f"Body text.\n\n{BLOCK}")

    def test_middle_splits_paragraphs_then_sentences_then_chars(self) -> None:
        # "Paragraphs" are 800-char chunks, so short text falls through to the sentence split.
        self.assertEqual(place_ad("P1.\n\nP2.\n\nP3.", BLOCK, "middle"), f"P1.\n\n{BLOCK}\n\nP2. P3.")
        p1, p2, p3 = ("a" * 600, "b" * 600, "c" * 600)
        self.assertEqual(place_ad(f"{p1}\n\n{p2}\n\n{p3}", BLOCK, "middle"), f"{p1}\n\n{BLOCK}\n\n{p2}\n\n{p3}")
        self.assertEqual(place_ad("One. Two. Three. Four.", BLOCK, "middle"), f"One. Two.\n\n{BLOCK}\n\nThree. Four.")
        self.assertEqual(place_ad("abcd", BLOCK, "middle"), f"ab\n\n{BLOCK}\n\ncd")

    def test_semantic_and_errors(self) -> None:
        chunks = ["Alpha.", "Beta.", "Gamma."]
        self.assertEqual(
            place_ad("ignored", BLOCK, "semantic", chunks=chunks, best_chunk_index=1),
            f"Alpha.\n\nBeta.\n\n{BLOCK}\n\nGamma.",
        )
        with self.assertRaises(ValueError):
            place_ad("x", BLOCK, "semantic")
        with self.assertRaises(ValueError):
            place_ad("x", BLOCK, "nowhere")


if __name__ == "__main__":
    unittest.main()
