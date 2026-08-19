from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARENA_OUTPUT_DIR = REPO_ROOT / "data/processed/lmarena/dataset"
DEFAULT_QUERIES_INPUT = DEFAULT_ARENA_OUTPUT_DIR / "arena_preference_en_single_turn.parquet"
DEFAULT_CATEGORIES_OUTPUT = REPO_ROOT / "data/processed/lmarena/lmarena_query_qwen3_categories.csv"
DEFAULT_DOMAINS_TXT = REPO_ROOT / "data/processed/lmarena/domains.txt"
DEFAULT_ADS_JSONL = REPO_ROOT / "data/processed/ads/ads2.jsonl"

QUERY_COLUMN = "query"
CATEGORY_COLUMNS = ["domain", "intent", "commercial_intent"]
QUERY_LOG_CHARS = 120

CATEGORY_MODEL_NAME = "Qwen/Qwen3-32B"
CATEGORY_MAX_NEW_TOKENS = 256
CATEGORY_MAX_RETRIES = 3
CATEGORY_BATCH_SIZE = 32
CATEGORY_MAX_MODEL_LEN = 32768
CATEGORY_GPU_MEMORY_UTILIZATION = 0.90

AD_MODEL_NAME = "nvidia/Llama-3.3-70B-Instruct-FP8"
AD_MAX_MODEL_LEN = 4096
AD_MAX_TOKENS = 1000
AD_GPU_MEMORY_UTILIZATION = 0.95
ADS_PER_DOMAIN = 2

ARENA_DATASET_ID = "lmarena-ai/arena-human-preference-140k"
ARENA_KEEP_COLUMNS = ["id", "model", "query", "llm_response", "timestamp"]
