from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ADS_JSONL = REPO_ROOT / "data/processed/ads/ads2.jsonl"
DEFAULT_INDEX_DIR = REPO_ROOT / "data/processed/ads/index"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 5
ENCODE_BATCH_SIZE = 32
INDEX_FILENAME = "index.faiss"
ADS_SIDECAR_FILENAME = "ads.json"
