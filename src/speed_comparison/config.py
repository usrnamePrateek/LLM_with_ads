"""Configuration variables specific to the speed comparison benchmarking module."""
from pathlib import Path

from src.common.shared_config import REPO_ROOT

DEFAULT_QUERIES_PARQUET = REPO_ROOT / "data/processed/lmarena/dataset/arena_preference_en_single_turn.parquet"
DEFAULT_INDEX_DIR = REPO_ROOT / "data/processed/ads/index_new"
DEFAULT_SPEED_COMPARISON_DIR = REPO_ROOT / "data/processed/speed_comparison"
DEFAULT_OUTPUT_CSV = DEFAULT_SPEED_COMPARISON_DIR / "speed_comparison_results.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_SPEED_COMPARISON_DIR / "speed_comparison_summary.json"

DEFAULT_NUM_QUERIES = 1000
DEFAULT_TOP_K = 1
ID_COLUMN = "id"
QUERY_COLUMN = "query"

# Embedding retrieval config
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RETRIEVAL_DEVICE = "cuda"

# LLM generation config
DEFAULT_LLM_MODEL = "nvidia/Llama-3.3-70B-Instruct-FP8"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BATCH_SIZE = 1
CHECKPOINT_EVERY = 50
