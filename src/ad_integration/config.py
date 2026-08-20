"""Configuration variables for placing ads into generated LLM responses."""
from pathlib import Path

from src.common.shared_config import (
    DEFAULT_INDEX_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_PLACED_ADS_CSV,
    DEFAULT_QUERIES_INPUT,
    REPO_ROOT,
)

DEFAULT_QUERIES_PARQUET = REPO_ROOT / "data/processed/lmarena/dataset/arena_preference_en_single_turn.parquet"
DEFAULT_SEMANTIC_PARAS_CSV = (
    REPO_ROOT / "data/processed/generation/query_semantic_paragraphs.csv"
)
QUERY_COLUMN = "query"
ID_COLUMN = "id"
LLM_RESPONSE_COLUMN = "llm_response"
AD_POSITIONS = ("first", "middle", "last", "semantic")
TOP_K = 1
ENCODE_BATCH_SIZE = 32
CHUNK_SIZE = 800
CHUNK_OVERLAP = 0

