"""Tests for speed comparison summary helpers (imports torch, but runs on CPU without vLLM)."""
import dataclasses
import tempfile
import unittest
from pathlib import Path

from src.speed_comparison.entities import QueryRetrievalTiming, QuerySpeedComparisonRecord
from src.speed_comparison.repository import SpeedComparisonRepository
from src.speed_comparison.services import _records_from_frame, _retrieval_only_summary


def _record(i: int) -> QuerySpeedComparisonRecord:
    return QuerySpeedComparisonRecord(
        id=str(i), query=f"q{i}", retrieval_time_ms=10.0 + i, embed_time_ms=8.0, search_time_ms=2.0 + i,
        retrieved_ad_id="A::1", retrieved_ad_score=0.5, llm_time_ms=1000.0 * i, prompt_tokens=5,
        output_tokens=50, tokens_per_sec=50.0, speedup_ratio=100.0, time_delta_ms=990.0,
        retrieval_finished_first=True, timestamp="ts",
    )


class SpeedSummaryHelperTests(unittest.TestCase):
    def test_records_round_trip_through_csv(self) -> None:
        records = [_record(1), _record(2)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            repo = SpeedComparisonRepository()
            repo.save_records(records, path)
            loaded = _records_from_frame(repo.load_records(path))
        self.assertEqual(loaded, [dataclasses.replace(r, timestamp="") for r in records])

    def test_retrieval_only_summary(self) -> None:
        timings = [
            QueryRetrievalTiming(id=str(i), query="q", embed_time_ms=8.0, search_time_ms=float(i),
                                 total_retrieval_ms=8.0 + i, retrieved_ad_id="A::1", retrieved_ad_score=0.5)
            for i in (1, 3)
        ]
        summary = _retrieval_only_summary(timings, "cpu", 2.5)
        self.assertEqual(summary.sample_size, 2)
        self.assertEqual(summary.llm_model, "(skipped)")
        self.assertEqual(summary.retrieval_distribution.mean, 10.0)
        self.assertEqual(summary.search_distribution.max, 3.0)
        self.assertEqual(summary.llm_distribution.mean, 0.0)
        self.assertEqual(summary.pct_retrieval_finished_first, 100.0)
        self.assertEqual(summary.total_retrieval_wall_clock_sec, 2.5)


if __name__ == "__main__":
    unittest.main()
