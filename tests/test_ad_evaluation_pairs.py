"""Characterization tests for the pairwise judge services (pair building, resume, parse failures)."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ad_evaluation.repository import PlacementComparisonScoreCsvRepository, PreferenceScoreCsvRepository
from src.ad_evaluation.services import EvaluateAdPreferenceService, EvaluatePlacementComparisonService
from src.common.io import CheckpointError

ADS = [
    {"domain": "A", "ad_id": 1, "headline": "a1", "description": "d", "cta": "c", "category": "cat_a"},
    {"domain": "A", "ad_id": 2, "headline": "a2", "description": "d", "cta": "c", "category": "cat_a"},
    {"domain": "A", "ad_id": 3, "headline": "a3", "description": "d", "cta": "c", "category": "cat_a"},
    {"domain": "B", "ad_id": 1, "headline": "b1", "description": "d", "cta": "c", "category": "cat_b"},
    {"domain": "B", "ad_id": 2, "headline": "b2", "description": "d", "cta": "c", "category": "cat_b"},
    {"domain": "C", "ad_id": 1, "headline": "c1", "description": "d", "cta": "c"},
]

GOOD_PREFERENCE = '{"winner_ad": "ad_1", "confidence": "high"}'
GOOD_PLACEMENT = '{"winner_pos": "pos_2", "confidence": "low"}'


class StubJudge:
    """Returns a valid answer for every prompt except the ones at `bad_indexes` (counted across calls)."""

    def __init__(self, good: str, bad_indexes: set[int] = frozenset()) -> None:
        self._good = good
        self._bad = bad_indexes
        self._seen = 0
        self.prompts: list[str] = []

    def generate(self, prompts: list[str]) -> list[str]:
        out = []
        for prompt in prompts:
            out.append("not json" if self._seen in self._bad else self._good)
            self.prompts.append(prompt)
            self._seen += 1
        return out


def _make_preference_service(mode: str, judge: StubJudge):
    return EvaluateAdPreferenceService(writer=PreferenceScoreCsvRepository(), judge=judge, candidates=mode)


class PreferencePairTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.ads_path = root / "ads.jsonl"
        self.ads_path.write_text("\n".join(json.dumps(ad) for ad in ADS), encoding="utf-8")
        self.assignments_path = root / "assignments.csv"
        pd.DataFrame(
            [
                {"id": "q1", "query": "first query", "ad_id": "A::1"},
                {"id": "q2", "query": "no category", "ad_id": "C::1"},
                {"id": "q3", "query": "unknown ad", "ad_id": "Z::9"},
            ]
        ).to_csv(self.assignments_path, index=False)
        self.output_path = root / "out.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, mode: str, judge: StubJudge, batch_size: int = 100):
        service = _make_preference_service(mode, judge)
        return service.run(
            assignments_path=self.assignments_path,
            ads_path=self.ads_path,
            output_path=self.output_path,
            batch_size=batch_size,
        )

    @staticmethod
    def _pairs(rows) -> list[tuple]:
        return [(r.query_id, r.ad_1_id, r.ad_2_id, r.is_swapped) for r in rows]

    def test_same_category_pairs_in_order(self) -> None:
        rows = self._run("same_category", StubJudge(GOOD_PREFERENCE), batch_size=3)
        self.assertEqual(
            self._pairs(rows),
            [
                ("q1", "A::1", "A::2", False),
                ("q1", "A::2", "A::1", True),
                ("q1", "A::1", "A::3", False),
                ("q1", "A::3", "A::1", True),
            ],
        )
        self.assertTrue(all(r.winner_ad_id == r.ad_1_id and r.confidence == "high" for r in rows))
        saved = pd.read_csv(self.output_path, dtype=str)
        self.assertEqual(len(saved), 4)
        self.assertEqual(
            list(saved.columns),
            ["query_id", "query", "ad_1_id", "ad_2_id", "winner_ad_id", "confidence", "is_swapped"],
        )

    def test_cross_category_pairs_in_order(self) -> None:
        rows = self._run("cross_category", StubJudge(GOOD_PREFERENCE))
        self.assertEqual(
            self._pairs(rows),
            [
                ("q1", "A::1", "B::1", False),
                ("q1", "B::1", "A::1", True),
                ("q1", "A::1", "B::2", False),
                ("q1", "B::2", "A::1", True),
            ],
        )

    def test_resume_skips_scored_pairs_and_appends(self) -> None:
        self._run("same_category", StubJudge(GOOD_PREFERENCE))
        saved = pd.read_csv(self.output_path, dtype=str)
        saved.head(1).to_csv(self.output_path, index=False)

        rows = self._run("same_category", StubJudge(GOOD_PREFERENCE))
        self.assertEqual(
            self._pairs(rows),
            [
                ("q1", "A::2", "A::1", True),
                ("q1", "A::1", "A::3", False),
                ("q1", "A::3", "A::1", True),
            ],
        )
        self.assertEqual(len(pd.read_csv(self.output_path)), 4)

    def test_damaged_existing_output_stops_the_run_instead_of_overwriting(self) -> None:
        self._run("same_category", StubJudge(GOOD_PREFERENCE))
        saved = pd.read_csv(self.output_path).rename(columns={"ad_1_id": "Ad 1"})
        saved.to_csv(self.output_path, index=False)
        before = self.output_path.read_bytes()
        judge = StubJudge(GOOD_PREFERENCE)
        with self.assertRaises(CheckpointError):
            self._run("same_category", judge)
        self.assertEqual(self.output_path.read_bytes(), before)
        self.assertEqual(judge.prompts, [])

    def test_parse_failure_is_not_saved_and_retried_on_resume(self) -> None:
        rows = self._run("same_category", StubJudge(GOOD_PREFERENCE, bad_indexes={1}))
        self.assertEqual(self._pairs(rows), [("q1", "A::1", "A::2", False), ("q1", "A::1", "A::3", False), ("q1", "A::3", "A::1", True)])
        self.assertNotIn("ERROR", set(pd.read_csv(self.output_path, dtype=str)["winner_ad_id"]))

        judge = StubJudge(GOOD_PREFERENCE)
        rows = self._run("same_category", judge)
        self.assertEqual(self._pairs(rows), [("q1", "A::2", "A::1", True)])
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual(len(pd.read_csv(self.output_path)), 4)


class PlacementComparisonPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        block = "---\n\n# h\n\n### d\n\n### [c](#)\n\n---"
        positions = [
            {"id": "q1", "query": "query one", "ad_id": "A::1", "position": pos, "response_with_ad": f"{pos} text\n\n{block}"}
            for pos in ("first", "middle", "last", "semantic")
        ] + [
            {"id": "q2", "query": "no semantic", "ad_id": "A::2", "position": "first", "response_with_ad": block},
        ]
        self.positions_path = root / "positions.csv"
        pd.DataFrame(positions).to_csv(self.positions_path, index=False)
        self.ads_path = root / "top_ads.csv"
        pd.DataFrame(
            [
                {"id": "q1", "ad_id": "A::1", "headline": "h", "description": "d", "cta": "c"},
                {"id": "q2", "ad_id": "A::2", "headline": "h", "description": "d", "cta": "c"},
            ]
        ).to_csv(self.ads_path, index=False)
        self.output_path = root / "out.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, judge: StubJudge, limit: int | None = None):
        service = EvaluatePlacementComparisonService(writer=PlacementComparisonScoreCsvRepository(), judge=judge)
        return service.run(
            positions_path=self.positions_path,
            ads_path=self.ads_path,
            output_path=self.output_path,
            limit=limit,
        )

    def test_pairs_resume_and_winner(self) -> None:
        judge = StubJudge(GOOD_PLACEMENT, bad_indexes={5})
        rows = self._run(judge)
        self.assertEqual(
            [(r.query_id, r.pos_1, r.pos_2, r.is_swapped) for r in rows],
            [
                ("q1", "semantic", "first", False),
                ("q1", "first", "semantic", True),
                ("q1", "semantic", "middle", False),
                ("q1", "middle", "semantic", True),
                ("q1", "semantic", "last", False),
            ],
        )
        self.assertEqual(rows[0].winner_pos, "first")
        self.assertTrue(all("[ad slot]" in prompt for prompt in judge.prompts))
        self.assertEqual(len(pd.read_csv(self.output_path)), 5)

        # The unparsed pair is asked again on resume; nothing else is re-judged.
        rows = self._run(StubJudge(GOOD_PLACEMENT))
        self.assertEqual([(r.pos_1, r.pos_2, r.is_swapped) for r in rows], [("last", "semantic", True)])
        self.assertEqual(len(pd.read_csv(self.output_path)), 6)

        saved = pd.read_csv(self.output_path, dtype=str)
        saved.head(2).to_csv(self.output_path, index=False)
        rows = self._run(StubJudge(GOOD_PLACEMENT), limit=1)
        self.assertEqual([(r.pos_1, r.pos_2, r.is_swapped) for r in rows], [("semantic", "middle", False)])
        self.assertEqual(len(pd.read_csv(self.output_path)), 3)


if __name__ == "__main__":
    unittest.main()
