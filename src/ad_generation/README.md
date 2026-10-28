# `ad_generation`

Prepares the LMArena dataset and generates the synthetic ad catalog.

## Current pipeline

| Script | What it does | Output |
|---|---|---|
| `scripts/prepare_dataset.py` | Downloads `lmarena-ai/arena-human-preference-140k`, keeps English single-turn conversations, picks the winning response | `data/processed/lmarena/dataset/arena_preference_en_single_turn.parquet` |
| `scripts/generate_bulk_ads.py` | Generates ad creatives (headline, description, CTA) per taxonomy sub-topic with Llama-3.3-70B; resumable, appends | `data/processed/ads_index/ads.jsonl` |

Input for bulk ads: `data/processed/ad_topics/taxonomy_subtopics.csv`, exported by
`analysis/ads_domain/new/google_topics_api.ipynb` from the Topics API taxonomy.
The ad `category` field (the taxonomy topic) is what `ad_evaluation` groups ads by.

## Legacy: `old/`

`old/` holds the earlier **query categorization** step: Qwen3-32B labels each query with
`domain`, `intent` and `commercial_intent` (0–3). It is no longer part of the pipeline; ads are now
generated per taxonomy sub-topic instead, and nothing downstream reads the categories CSV.

- `old/generate_categories.py`: script (`./run_vllm.sh -m src.ad_generation.old.generate_categories`)
- `old/categories.py`: its config, entity, prompt/schema, parser, checkpoint repository and service
- `old/category_generator.py`: the vLLM generator

Output (kept for reference): `data/processed/lmarena/old/lmarena_query_qwen3_categories.csv`.
The exploration notebook is `analysis/ads_domain/old/categories.ipynb`.
Do not add new dependencies on `old/`; it is kept only to reproduce the original categories.
