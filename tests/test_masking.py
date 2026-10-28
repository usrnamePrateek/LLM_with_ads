"""Tests for strict ad masking and how the judge services handle responses that cannot be masked."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ad_evaluation.core.masking import AdBlockNotFoundError, mask_ad_in_response
from src.ad_evaluation.repository import (
    PlacementComparisonScoreCsvRepository,
    PlacementInputRepository,
    PlacementScoreCsvRepository,
)
from src.ad_evaluation.services import EvaluatePlacementComparisonService, ScorePlacementsService
from src.ad_integration.core.placement import format_ad_block, place_ad

BLOCK = format_ad_block("Try Acme VPN", "Fast and private browsing.", "")


class StubJudge:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.prompts: list[str] = []

    def generate(self, prompts: list[str]) -> list[str]:
        self.prompts.extend(prompts)
        return [self._answer] * len(prompts)


class MaskTests(unittest.TestCase):
    def test_exact_block_is_replaced(self) -> None:
        response = place_ad("Answer text.", BLOCK, "last")
        masked, ad_text = mask_ad_in_response(response, "Try Acme VPN", "Fast and private browsing.", "")
        self.assertEqual(masked, "Answer text.\n\n[ad slot]")
        self.assertEqual(ad_text, BLOCK)

    def test_mismatch_raises_instead_of_cutting_answer_text(self) -> None:
        # The answer has its own ---/#/### markdown; the old regex fallback deleted "Part two." here.
        response = f"Part one.\n\n---\n\n# Heading\n\n### Sub\n\nPart two.\n\n{BLOCK}\n\nPart three."
        with self.assertRaises(AdBlockNotFoundError):
            mask_ad_in_response(response, "Try Acme VPN", "Fast and private browsing.", "nan")
        with self.assertRaises(AdBlockNotFoundError):
            mask_ad_in_response("No ad here.", "Try Acme VPN", "Fast and private browsing.", "")


class ServiceMaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ads_path = self.root / "top_ads.csv"
        pd.DataFrame(
            [
                {"id": "q1", "ad_id": "A::1", "headline": "Try Acme VPN", "description": "Fast and private browsing.", "cta": None},
                {"id": "q2", "ad_id": "A::2", "headline": "Other", "description": "Different ad.", "cta": "Go"},
            ]
        ).to_csv(self.ads_path, index=False)
        rows = [
            {"id": "q1", "query": "vpn?", "ad_id": "A::1", "position": pos, "response_with_ad": place_ad("Answer.", BLOCK, pos if pos != "semantic" else "last")}
            for pos in ("first", "middle", "last", "semantic")
        ]
        # q2's responses contain no ad block at all, so they cannot be masked.
        rows += [
            {"id": "q2", "query": "other?", "ad_id": "A::2", "position": pos, "response_with_ad": "Plain answer."}
            for pos in ("first", "semantic")
        ]
        self.positions_path = self.root / "positions.csv"
        pd.DataFrame(rows).to_csv(self.positions_path, index=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_score_placements_skips_unmaskable_rows(self) -> None:
        judge = StubJudge('{"score": 4, "reason": "fine"}')
        output = self.root / "scores.csv"
        rows = ScorePlacementsService(PlacementInputRepository(), PlacementScoreCsvRepository(), judge).run(
            self.positions_path, self.ads_path, output_path=output
        )
        self.assertEqual([(r.id, r.position) for r in rows], [("q1", p) for p in ("first", "middle", "last", "semantic")])
        self.assertTrue(all("Try Acme VPN" not in p.split("[ad slot]")[0] for p in judge.prompts))
        self.assertEqual(len(pd.read_csv(output)), 4)

    def test_score_placements_does_not_save_unparsed_answers(self) -> None:
        output = self.root / "scores.csv"
        service = lambda judge: ScorePlacementsService(PlacementInputRepository(), PlacementScoreCsvRepository(), judge)
        self.assertEqual(service(StubJudge("not json")).run(self.positions_path, self.ads_path, output_path=output), [])
        self.assertFalse(output.exists())

        rows = service(StubJudge('{"score": 4, "reason": "fine"}')).run(self.positions_path, self.ads_path, output_path=output)
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(pd.read_csv(output)["score"]), {4})

    def test_placement_comparison_masks_empty_cta_and_skips_unmaskable(self) -> None:
        judge = StubJudge('{"winner_pos": "pos_1", "confidence": "high"}')
        rows = EvaluatePlacementComparisonService(PlacementComparisonScoreCsvRepository(), judge).run(
            self.positions_path, self.ads_path, output_path=self.root / "cmp.csv"
        )
        # Empty CTA used to become "nan" and miss the exact match; now all 6 q1 pairs are built, q2 is skipped.
        self.assertEqual(len(rows), 6)
        self.assertEqual({r.query_id for r in rows}, {"q1"})
        self.assertTrue(all(p.count("[ad slot]") == 2 for p in judge.prompts))


if __name__ == "__main__":
    unittest.main()
