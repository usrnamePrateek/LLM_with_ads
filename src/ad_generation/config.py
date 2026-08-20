"""Configuration variables specific to the ad generation and data preparation module."""
from pathlib import Path
from src.common.shared_config import (
    AD_GPU_MEMORY_UTILIZATION,
    AD_MAX_MODEL_LEN,
    AD_MAX_TOKENS,
    AD_MODEL_NAME,
    ARENA_DATASET_ID,
    ARENA_KEEP_COLUMNS,
    DEFAULT_ADS_JSONL,
    DEFAULT_ARENA_OUTPUT_DIR,
    DEFAULT_QUERIES_INPUT,
    REPO_ROOT,
)
DEFAULT_CATEGORIES_OUTPUT = REPO_ROOT / "data/processed/lmarena/lmarena_query_qwen3_categories.csv"
DEFAULT_DOMAINS_TXT = REPO_ROOT / "data/processed/lmarena/domains.txt"

QUERY_COLUMN = "query"
CATEGORY_COLUMNS = ["domain", "intent", "commercial_intent"]
QUERY_LOG_CHARS = 120

CATEGORY_MODEL_NAME = "Qwen/Qwen3-32B"
CATEGORY_MAX_NEW_TOKENS = 256
CATEGORY_MAX_RETRIES = 3
CATEGORY_BATCH_SIZE = 32
CATEGORY_MAX_MODEL_LEN = 32768
CATEGORY_GPU_MEMORY_UTILIZATION = 0.90

ADS_PER_DOMAIN = 2

