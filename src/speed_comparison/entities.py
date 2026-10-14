"""Domain entities and dataclasses for the speed comparison benchmarking module."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QueryRetrievalTiming:
    """Timing and result breakdown for retrieving an ad for a single query."""
    id: str
    query: str
    embed_time_ms: float
    search_time_ms: float
    total_retrieval_ms: float
    retrieved_ad_id: str
    retrieved_ad_score: float
    timestamp: str = ""


@dataclass
class QueryLlmTiming:
    """Timing and token metrics for generating an LLM response for a single query."""
    id: str
    query: str
    llm_time_ms: float
    prompt_tokens: int
    output_tokens: int
    tokens_per_sec: float
    llm_response: str
    timestamp: str = ""


@dataclass
class QuerySpeedComparisonRecord:
    """Combined comparison record for a single query comparing retrieval and LLM generation."""
    id: str
    query: str
    # Ad retrieval metrics
    retrieval_time_ms: float
    embed_time_ms: float
    search_time_ms: float
    retrieved_ad_id: str
    retrieved_ad_score: float
    # LLM generation metrics
    llm_time_ms: float
    prompt_tokens: int
    output_tokens: int
    tokens_per_sec: float
    # Comparative metrics
    speedup_ratio: float  # llm_time_ms / retrieval_time_ms
    time_delta_ms: float  # llm_time_ms - retrieval_time_ms
    retrieval_finished_first: bool
    timestamp: str = ""


@dataclass
class LatencyDistribution:
    """Statistical summary of a latency metric distribution in milliseconds."""
    mean: float
    std: float
    min: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float


@dataclass
class SpeedBenchmarkSummary:
    """High-level summary of the speed comparison benchmark across all evaluated queries."""
    sample_size: int
    llm_model: str
    embedding_model: str
    retrieval_device: str
    retrieval_distribution: LatencyDistribution
    embed_distribution: LatencyDistribution
    search_distribution: LatencyDistribution
    llm_distribution: LatencyDistribution
    mean_speedup_ratio: float
    median_speedup_ratio: float
    mean_time_delta_ms: float
    pct_retrieval_finished_first: float
    mean_tokens_per_sec: float
    total_retrieval_wall_clock_sec: float
    total_llm_wall_clock_sec: float
    timestamp: str = ""


def compute_distribution(values: list[float]) -> LatencyDistribution:
    """Computes mean, standard deviation, and percentiles for a list of values."""
    if not values:
        return LatencyDistribution(
            mean=0.0, std=0.0, min=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, max=0.0
        )
    arr = np.asarray(values, dtype=float)
    return LatencyDistribution(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max=float(np.max(arr)),
    )
