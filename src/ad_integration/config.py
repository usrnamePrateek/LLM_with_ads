"""Configuration variables for placing ads into generated LLM responses."""
from src.common.shared_config import REPO_ROOT

DEFAULT_SEMANTIC_PARAS_CSV = (
    REPO_ROOT / "data/processed/ad_integration/bulk_ads/query_semantic_paragraphs.csv"
)
QUERY_COLUMN = "query"
ID_COLUMN = "id"
LLM_RESPONSE_COLUMN = "llm_response"
AD_POSITIONS = ("first", "middle", "last", "semantic")
TOP_K = 1
CHUNK_SIZE = 800
CHUNK_OVERLAP = 0

