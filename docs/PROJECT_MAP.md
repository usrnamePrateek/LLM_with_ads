# Project Map

Orientation doc for future sessions. Built from static reading only (nothing was run).
Items marked (unverified) were inferred, not confirmed. See also `docs/architecture.md`
`AGENTS.md`, `.agents/rules/python.md`.

## What it does
Research pipeline that measures how ads can be inserted into LLM answers and how natural
the placement reads. Source data: LMArena human-preference dataset
(`lmarena-ai/arena-human-preference-140k`). Steps: generate synthetic ads -> embed and index
them -> retrieve the top ad per query -> insert it into the stored LLM response at 4
positions -> LLM-as-judge scores placements and compares ads/positions pairwise.
A side module benchmarks retrieval latency vs LLM generation latency.
User: the repo owner (research / thesis-style work; analysis in notebooks). Single GPU box.

## Directory map
- `src/common/` shared constants (`shared_config.py`) and vLLM base class (`shared_llm.py`)
- `src/ad_generation/` dataset prep, query categorization, ad generation, prompts, JSON parsing
- `src/ad_indexing/` BGE-M3 encoder, FAISS store, index build / query CLI
- `src/ad_integration/` top-ad assignment and ad placement into responses
- `src/ad_evaluation/` LLM judges (placement score, ad preference, cross-category, placement comparison)
- `src/speed_comparison/` latency benchmark (has own `AGENTS.md`)
- `tests/` one file: `test_speed_comparison.py` (entities + repository only)
- `analysis/` 5 notebooks (large outputs, do not read). `data/` all inputs/outputs (do not read)
- `docs/architecture.md`, `.agents/rules/python.md` (coding standards), `README.md`
- `.lmarena-env` symlink to a venv OUTSIDE the repo (`.../MTP/llm_with_ads/.venv`)

Every module uses the same layout: `config.py`, `entities.py` (frozen dataclasses),
`repository.py` (CSV/JSONL/parquet I/O), `services.py` (orchestration), `scripts/` (argparse CLIs).
All paths below are under `src/` unless noted.

## Data flow (files are the interfaces; paths under `data/processed/`)
1. `ad_generation/scripts/prepare_dataset.py` -> `PrepareArenaDatasetService.run`
   -> `lmarena/dataset/arena_preference_en_single_turn.parquet` (id, model, query, llm_response, timestamp)
2. `generate_categories.py` -> `CategorizeQueriesService` + `VllmCategoryGenerator` (Qwen3-32B)
   -> `lmarena/lmarena_query_qwen3_categories.csv` (+ domain, intent, commercial_intent)
3. Ads:
   `generate_bulk_ads.py` -> `GenerateBulkAdsService` + `VllmBulkAdGenerator`: reads `lmarena/taxonomy_subtopics.csv`,
   `--target-ads` default 1000, APPENDS to `ads/ads.jsonl`, resumes from per-domain counts
   (the older 2-ads-per-domain `generate_ads` route was removed)
   - ad record: domain, ad_id, headline, description, cta, category, subtopic
4. `ad_indexing/scripts/build_index.py` -> `IndexAdsService` -> `ads/index/index.faiss` + `ads.json` sidecar.
   Ad id is `"{domain}::{ad_id}"` (`entities.make_ad_id`); embed text = headline + "\n" + description.
5. `ad_integration/scripts/assign_top_ads.py` -> `AssignTopAdService`: encode queries, `store.query`, take hit 0
   -> `generation/bulk_ads/query_top1_ads.csv` (id, query, ad_id, ad_domain, headline, description, cta, score). Resumes by `id`.
   All integration/evaluation defaults now live under `generation/bulk_ads/` (`generation/` itself is legacy).
6. `place_ads.py` -> `PlaceAdsInResponsesService`: inner-join dataset and assignments on `id`; 4 rows per query
   -> `generation/bulk_ads/query_ad_positions.csv` (id, query, ad_id, position, response_with_ad) and
   `generation/bulk_ads/query_semantic_paragraphs.csv`. NO resume: it deletes both outputs first.
7. `ad_evaluation/scripts/`:
   - `score_placements.py` -> `ScorePlacementsService` + `VllmPlacementJudge` -> `ad_evaluation/placement_scores.csv` (score 1-5, reason)
   - `run_preference_eval.py` -> ad vs other ads in same category, both orders -> `data/evaluation/preference_scores.csv`
   - `run_cross_category_eval.py` -> vs ads in other categories -> `data/evaluation/cross_category_preference_scores.csv`
   - `run_placement_comparison.py` -> semantic vs first/middle/last, both orders -> `data/evaluation/placement_comparison_scores.csv`
8. `speed_comparison/scripts/run_speed_comparison.py` -> `SpeedComparisonOrchestrator.run` -> `speed_comparison/run_<ts>/`
   Phases: retrieval (encoder + FAISS timed separately, sequential), free encoder, LLM generation one query at a time, summary
   (mean/std/min/p50/p90/p95/p99/max, speedup). `--retrieval-only --retrieval-device cpu` needs no vLLM/GPU.

### Placement (`ad_integration/core/placement.py`, `semantic.py`, `splitting.py`)
Ad block is markdown: `---`, `# headline`, `### description`, `### [cta](#)`, `---`.
`first` / `last` prepend/append; `middle` splits at len//2 (paragraph, then sentence, then char fallback);
`semantic` inserts after the chunk with the highest cosine to the ad vector (`score_chunks`,
`insert_ad_after_best_paragraph`). "Paragraphs" are really 800-char LangChain chunks (CHUNK_SIZE=800).

### Judges (`ad_evaluation/llm/`)
`prompts.py` (system prompts + JSON schemas), `judge.py` / `preference_judge.py` (vLLM classes on `BaseVllmGenerator`),
`parsing.py` (strict parsers). `core/masking.py::mask_ad_in_response` replaces the ad block with `[ad slot]`
before judging. Placement comparison uses a DeepSeek-R1-Distill model with free-form reasoning then a ```json block.

## Module dependencies
- `common.shared_llm` -> `ad_generation.llm.parsing` (strip_thinking): common depends on a feature module (acknowledged in `common/AGENTS.md`)
- `ad_generation.config`, `ad_indexing.config`, `ad_integration.config`, `speed_comparison.config` -> `common.shared_config`
- `ad_integration` -> `ad_indexing` (encoder, FAISS store)
- `ad_evaluation` -> `ad_integration.core.placement`, `ad_indexing.repository`, `ad_generation.llm.{parsing,pathutil}`, `common`
- `speed_comparison` -> `ad_integration.QueryDatasetRepository`, `ad_indexing`, `common.shared_llm`
- `ad_indexing` and `ad_generation` are coupled only through `ads.jsonl`
- Do not touch FAISS outside `ad_indexing`; use the `VectorStore` / `EmbeddingEncoder` protocols

## Models and key config
| Use | Value |
|---|---|
| Ad gen + placement judge | `nvidia/Llama-3.3-70B-Instruct-FP8` (ads: len 4096, 1000 tok, gpu 0.95; judge: len 16384, 256 tok, batch 128, checkpoint 1000) |
| Categorization | `Qwen/Qwen3-32B` (batch 32, len 32768, 3 retries, gpu 0.90) |
| Placement comparison judge | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (len 131072, 4096 tok, batch 32, fp8 hardcoded) |
| Embeddings | `BAAI/bge-m3`, `IndexFlatIP`, top-k 5 (indexing default) / 1 (integration) |
| Speed bench | 1000 queries, 512 new tokens, temp 0, index at `ads/index_new` |
Ad generation sampling: temp 0.7, top_p 0.8, top_k 20.

## External services, env vars, hardware
- Hugging Face (dataset + model download): `HF_TOKEN` (read in `ad_generation/services.py` via dotenv)
- `.env.example` / CLAUDE.md list only `HF_TOKEN` (`GEMINI_API_KEY` was dropped as unused). `google-genai` is still in `requirements.txt`; no use in `src/` (unverified for `analysis/google_topics_api.ipynb`)
- No DB. Everything is files. GPU required for vLLM and BGE-M3 (CPU encoder path exists only for speed benchmark)
- `ad_generation/llm/pathutil.prepend_venv_bin_to_path` edits PATH so `ninja` is found (JIT builds)

## Run, test, debug
- From repo root: `./run_vllm.sh -m src.<module>.scripts.<name> [args]` (`unset LD_LIBRARY_PATH`, runs `.lmarena-env/bin/python`; relative path, so must run from root)
- Typical order: prepare_dataset -> generate_categories -> generate_(bulk_)ads -> build_index -> assign_top_ads -> place_ads -> evaluation scripts
- Most scripts accept `--limit N`; use it for smoke runs
- Tests: `./run_vllm.sh -m unittest discover -s tests -t .` (7 tests, speed_comparison only). No pytest/lint/typecheck configured
- Tests must not hit network/GPU. `services.py` in speed_comparison imports torch at top, so the tests avoid it
- Debug: failures are printed (not logged); parse failures become rows (`score=0`, winner "ERROR") or are skipped

## Conventions
- Checkpointing: append-mode CSV/JSONL, skip done IDs on restart (keys listed below)
- `dataclasses.asdict()` for output rows; no `to_dict()`
- Each `config.py` defines values directly (but see re-exports under tech debt)
- Services take injected repositories/generators/encoders; scripts wire them
- LLM generators subclass `common.shared_llm.BaseVllmGenerator` (guided JSON via `json_schema`, `strip_thinking` on output)
- `core/` functions (placement, splitting, masking) are pure; no I/O

Resume keys: assign_top_ads `id`; score_placements `(id, ad_id, position)`; preference/cross-category
`(query_id, ad_1_id, ad_2_id, is_swapped)`; placement comparison `(query_id, ad_id, pos_1, pos_2, is_swapped)`;
speed_comparison `id` (only with explicit `--run-dir`); categories merge by `id`.

## Fragile areas and tech debt
Correctness risks:
- Bulk generation appends to `ads.jsonl` and ad ids are `domain::ad_id`; hand-editing or double-appending the file can produce duplicate ids, and `JsonlAdRepository.load` raises on duplicates
- `FaissVectorStore.load` (`store/faiss_store.py:90`) rebuilds `Ad` without `category` (sidecar has it via `asdict(Ad)`, load ignores it). Low impact today: `ad_evaluation` reads categories from `ads.jsonl` via `JsonlAdRepository`, not from the loaded store
- `place_ads` has no resume and deletes outputs at start; a crash loses the run
- `mask_ad_in_response` returns the response UNMASKED if neither exact match nor regex matches, so the ad can leak to the judge
- Parse failures are saved as completed rows (score 0 / "ERROR"), so resume never retries them
- Resume code with missing columns or unreadable files can silently overwrite the existing output (assign_top_ads, preference services)
- speed_comparison: `--batch-size` is accepted but intentionally unused (reserved for future batching); default run creates a new `run_<ts>` dir so resume needs `--run-dir`; on resume the summary covers only new records; `--retrieval-only` ignores checkpoints
- assign_top_ads with no `id` column sets `id=""` for all rows; place_ads would then merge on empty ids (unverified impact)

Structure / hygiene:
- Import cycle-ish: `common` -> `ad_generation.llm.parsing`; `ad_evaluation` also reuses `ad_generation` parsing helpers
- Config re-exports across modules (`ad_integration.config`, `ad_indexing.config` re-export `common.shared_config`) contradict the "no derived config" convention; some constants unused (`ENCODE_BATCH_SIZE`)
- Copy-paste: the two preference services in `ad_evaluation/services.py` (527 lines); six identical CSV saver bodies
- `services.py` in `ad_generation` imports the vLLM wrapper at module top (torch pulled in for any import)
- String-literal type hints naming classes that are not imported; many function-local imports
- `ad_generation/fetch_taxonomy.py` is a scratch script (no main, nothing imports it)
- vllm/torch are intentionally not in `requirements.txt` (env is built outside the repo; how is not documented)
- CLI text drift: `run_placement_comparison` description says Llama 3.3 70B, default model is DeepSeek-R1-Distill
- Tests cover only speed_comparison entities/repository (`tests/__init__.py` added so `discover -t .` works)

## Where to look for X
- Shared paths and Arena constants: `common/shared_config.py`
- vLLM loading, chat template, truncation, thinking-strip: `common/shared_llm.py`
- Prompts and JSON schemas: `ad_generation/llm/prompts.py` (category + bulk ads), `ad_evaluation/llm/prompts.py`
- LLM output parsing: `ad_generation/llm/parsing.py`, `ad_evaluation/llm/parsing.py`
- Arena filtering (English, single-turn, winner pick): `ad_generation/arena.py`, `PrepareArenaDatasetService`
- Embeddings / vector search: `ad_indexing/embeddings/bge_m3.py`, `store/faiss_store.py`, `services.py`
- Where the ad goes in a response: `ad_integration/core/placement.py`, `core/semantic.py`
- Judge scoring logic and masking: `ad_evaluation/services.py`, `core/masking.py`
- Checkpoint/resume: each `services.py` (+ `repository.py` for checkpoint loaders)
- Latency stats: `speed_comparison/entities.py::compute_distribution`, `services.py`
- Coding standards: `.agents/rules/python.md`; module notes: `src/speed_comparison/AGENTS.md`, `src/common/AGENTS.md`
