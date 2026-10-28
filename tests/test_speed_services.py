"""Tests for speed comparison summary helpers (imports torch, but runs on CPU without vLLM)."""
import unittest

from src.speed_comparison.entities import QueryRetrievalTiming
from src.speed_comparison.services import _retrieval_only_summary


class SpeedSummaryHelperTests(unittest.TestCase):
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
