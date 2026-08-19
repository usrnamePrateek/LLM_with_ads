"""Shared configuration used across multiple modules.

Values that are consumed by more than one package (ad_retrieval, generation,
lmarena_prep, placement_testing) live here so that no package needs to import
another package's config.
"""

from pathlib import Path

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]

# ── LLM model settings (used by lmarena_prep + placement_testing) ────────────
AD_MODEL_NAME = "nvidia/Llama-3.3-70B-Instruct-FP8"
AD_MAX_MODEL_LEN = 4096
AD_MAX_TOKENS = 1000
AD_GPU_MEMORY_UTILIZATION = 0.95

# ── Data paths shared across modules ─────────────────────────────────────────
DEFAULT_ADS_JSONL = REPO_ROOT / "data/processed/ads/ads2.jsonl"
DEFAULT_INDEX_DIR = REPO_ROOT / "data/processed/ads/index"

DEFAULT_ARENA_OUTPUT_DIR = REPO_ROOT / "data/processed/lmarena/dataset"
DEFAULT_QUERIES_INPUT = DEFAULT_ARENA_OUTPUT_DIR / "arena_preference_en_single_turn.parquet"

DEFAULT_OUTPUT_CSV = REPO_ROOT / "data/processed/generation/query_top1_ads.csv"
DEFAULT_PLACED_ADS_CSV = REPO_ROOT / "data/processed/generation/query_ad_positions.csv"

# ── Arena dataset ────────────────────────────────────────────────────────────
ARENA_DATASET_ID = "lmarena-ai/arena-human-preference-140k"
ARENA_KEEP_COLUMNS = ["id", "model", "query", "llm_response", "timestamp"]
