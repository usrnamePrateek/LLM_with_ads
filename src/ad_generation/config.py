"""Configuration variables specific to the ad generation and data preparation module."""
from src.common.shared_config import (
    AD_GPU_MEMORY_UTILIZATION,
    AD_MAX_MODEL_LEN,
    AD_MAX_TOKENS,
    AD_MODEL_NAME,
    ARENA_DATASET_ID,
    ARENA_KEEP_COLUMNS,
    DEFAULT_ADS_JSONL,
    DEFAULT_ARENA_OUTPUT_DIR,
    REPO_ROOT,
)

TAXONOMY_CSV_PATH = REPO_ROOT / "data/processed/ad_topics/taxonomy_subtopics.csv"
MAX_ADS_PER_PROMPT = 5
