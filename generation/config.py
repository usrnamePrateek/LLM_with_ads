from pathlib import Path

from ad_retrieval.config import DEFAULT_INDEX_DIR
from lmarena_prep.config import DEFAULT_QUERIES_INPUT

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERIES_PARQUET = DEFAULT_QUERIES_INPUT
DEFAULT_INDEX_DIR = DEFAULT_INDEX_DIR
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data/processed/generation/query_top1_ads.csv"
QUERY_COLUMN = "query"
ID_COLUMN = "id"
TOP_K = 1
ENCODE_BATCH_SIZE = 32
