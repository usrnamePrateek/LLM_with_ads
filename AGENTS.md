# LLM Ad Placement Testing

## Pipeline

Sequential modules connected via filesystem (CSV/JSONL/Parquet):

1. **`src/ad_generation`** → Extract Arena dataset, categorize queries, generate synthetic ads.
2. **`src/ad_indexing`** → Embed ads with BGE-M3, build FAISS index.
3. **`src/ad_integration`** → Match queries to ads, inject ad blocks into LLM responses.
4. **`src/ad_evaluation`** → LLM-as-a-judge scores placement quality (1–5).
5. **`src/common`** → Shared config (`shared_config.py`) and vLLM base class (`shared_llm.py`).

## Development

- Virtual environment: `.lmarena-env`
- GPU required for vLLM and sentence-transformers inference.
- Consult `docs/architecture.md` for detailed data flow and module responsibilities.
- Consult `.agents/rules/python.md` for coding standards.

## Key Conventions

- **Checkpointing**: Long-running LLM batch scripts write incrementally (append mode) and skip already-processed IDs on restart.
- **Entity serialization**: Use `dataclasses.asdict()` for CSV/JSONL output. No custom `to_dict()` methods.
- **Config independence**: Each module's `config.py` should define values directly, not derive them from other config keys.
