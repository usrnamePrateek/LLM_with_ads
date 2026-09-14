"""Configuration variables for the LLM-as-a-judge placement evaluation pipeline."""

from src.common.shared_config import (
    AD_GPU_MEMORY_UTILIZATION,
    AD_MODEL_NAME,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_PLACED_ADS_CSV,
    REPO_ROOT,
)

DEFAULT_POSITIONS_CSV = REPO_ROOT / "data/processed/generation/query_ad_positions.csv"
DEFAULT_ADS_CSV = REPO_ROOT / "data/processed/generation/query_top1_ads.csv"
DEFAULT_SCORES_CSV = REPO_ROOT / "data/processed/ad_evaluation/placement_scores.csv"

JUDGE_MODEL_NAME = "nvidia/Llama-3.3-70B-Instruct-FP8"
JUDGE_MAX_MODEL_LEN = 16384
JUDGE_MAX_TOKENS = 256
JUDGE_GPU_MEMORY_UTILIZATION = 0.95
JUDGE_BATCH_SIZE = 128
CHECKPOINT_EVERY = 1000
AD_SLOT = "[ad slot]"

