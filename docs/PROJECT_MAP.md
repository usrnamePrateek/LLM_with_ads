# Project Map

The single reference for how the project is laid out, how data flows, the module rules, and fragile spots.
Coding standards live in `.agents/rules/python.md`; commands are in `CLAUDE.md`.
Items marked (unverified) were inferred, not confirmed.

## What it does
Research pipeline that measures how ads can be inserted into LLM answers and how natural
the placement reads. Source data: LMArena human-preference dataset
(`lmarena-ai/arena-human-preference-140k`). Steps: generate synthetic ads -> embed and index
them -> retrieve the top ad per query -> insert it into the stored LLM response at 4
positions -> LLM-as-judge scores placements and compares ads/positions pairwise.
A side module benchmarks retrieval latency vs LLM generation latency.
User: the repo owner (research / thesis-style work; analysis in notebooks). GPU work runs on an A100 80GB box.

## Directory map
- `src/common/` shared code with no business logic:
  `shared_config.py` (paths/models used by several modules), `shared_llm.py` (`BaseVllmGenerator`),
  `io.py` (`load_queries`, `save_rows`, `load_done_keys`), `text.py` (`strip_thinking`),
  `env.py` (`prepend_venv_bin_to_path`)
- `src/ad_generation/` dataset prep, ad generation, prompts, JSON parsing; `old/` holds the legacy query
  categorization step (see `src/ad_generation/README.md`)
- `src/ad_indexing/` BGE-M3 encoder, FAISS store, index build / query CLI
- `src/ad_integration/` top-ad assignment and ad placement into responses
- `src/ad_evaluation/` LLM judges (placement score, ad preference same/cross category, placement comparison)
- `src/speed_comparison/` latency benchmark + HTML report (`scripts/build_report.py`)
- `tests/` unittest, CPU only: characterization tests for evaluation pair building, CSV writing,
  query loading, placement, and speed_comparison entities/repository/summary helpers
- `analysis/` notebooks (large outputs, do not read); `analysis/ad_generation/` holds the dataset/category
  exploration notebooks. `data/` all inputs/outputs (do not read)
- `.lmarena-env` symlink to a venv OUTSIDE the repo (`.../MTP/llm_with_ads/.venv`)

Every module uses the same layout: `config.py`, `entities.py` (frozen dataclasses),
`repository.py` (CSV/JSONL/parquet I/O), `services.py` (orchestration), `scripts/` (argparse CLIs).
All paths below are under `src/` unless noted.

## Data flow (files are the interfaces; paths under `data/processed/`)
1. `ad_generation/scripts/prepare_dataset.py` -> `PrepareArenaDatasetService.run`
   -> `lmarena/dataset/arena_preference_en_single_turn.parquet` (id, model, query, llm_response, timestamp)
2. (legacy, not in the pipeline) `ad_generation/old/generate_categories.py` -> Qwen3-32B query categories
   -> `lmarena/old/lmarena_query_qwen3_categories.csv` (+ domain, intent, commercial_intent). Nothing downstream reads it.
3. `generate_bulk_ads.py` -> `GenerateBulkAdsService` + `VllmBulkAdGenerator`: reads `ad_topics/taxonomy_subtopics.csv`
   (exported by `analysis/ads_domain/new/google_topics_api.ipynb`),
   `--target-ads` default 1000, APPENDS to `ads_index/ads.jsonl`, resumes from per-domain counts
   - ad record: domain, ad_id, headline, description, cta, category, subtopic
4. `ad_indexing/scripts/build_index.py` -> `IndexAdsService` -> `ads_index/index_new/index.faiss` + `ads.json` sidecar
   (`DEFAULT_INDEX_DIR`; `ads_index/old/` holds the earlier 1,560-ad set and its index, unused).
   Ad id is `"{domain}::{ad_id}"` (`entities.make_ad_id`); embed text = headline + "\n" + description.
5. `ad_integration/scripts/assign_top_ads.py` -> `AssignTopAdService`: encode queries, `store.query`, take hit 0
   -> `ad_integration/bulk_ads/query_top1_ads.csv` (id, query, ad_id, ad_domain, headline, description, cta, score). Resumes by `id`.
   All integration/evaluation defaults live under `ad_integration/bulk_ads/` (`ad_integration/old/` holds legacy outputs).
6. `place_ads.py` -> `PlaceAdsInResponsesService`: inner-join dataset and assignments on `id`; 4 rows per query
   -> `ad_integration/bulk_ads/query_ad_positions.csv` (id, query, ad_id, position, response_with_ad) and
   `ad_integration/bulk_ads/query_semantic_paragraphs.csv`. NO resume: it deletes both outputs first.
7. `ad_evaluation/scripts/`:
   - `score_placements.py` -> `ScorePlacementsService` + `VllmPlacementJudge` -> `ad_evaluation/placement_scores.csv` (score 1-5, reason)
     (the earlier full run, 146,316 rows, is archived in `data/evaluation/ad_placement_eval/old/placement_scores.csv`; see its readme)
   - `run_preference_eval.py` / `run_cross_category_eval.py` -> one `EvaluateAdPreferenceService`
     with `candidates="same_category"` / `"cross_category"`; assigned ad vs each candidate ad, both orders
     -> `data/evaluation/ad_pref_eval/same_category_pref_scores.csv` / `.../ad_pref_eval/cross_category_preference_scores.csv`
   - `run_placement_comparison.py` -> semantic vs first/middle/last, both orders -> `data/evaluation/ad_placement_eval/placement_comparison_scores.csv`
8. `speed_comparison/scripts/run_speed_comparison.py` -> `SpeedComparisonOrchestrator.run` -> `speed_comparison/run_<ts>/`
   (see "speed_comparison" below). `build_report.py` -> `speed_comparison/reports/speed_comparison_report_<newest run ts>.html`
   (the `/speed-report` skill drives it).

## Module rules
- **common**: infrastructure only, no business logic, and it must not import feature modules.
  `BaseVllmGenerator` handles vLLM setup, chat template (`enable_thinking=False` fallback), truncation of user
  content to `max_model_len - max_new_tokens`, and `strip_thinking` on output. Changing truncation affects every
  generator (`VllmBulkAdGenerator`, the judges, `VllmBenchmarkGenerator`, legacy `old/VllmCategoryGenerator`).
- **ad_generation**: LLM parsing (`llm/parsing.py`) must tolerate malformed JSON; one bad row never crashes a batch.
  The ad `category` field (taxonomy topic) drives evaluation grouping. `old/` is legacy: do not add dependencies on it.
- **ad_indexing**: no placement or business logic. Other modules use the `VectorStore` / `EmbeddingEncoder`
  protocols (`store/base.py`, `embeddings/base.py`), never FAISS directly; import concrete classes from their
  submodules (`embeddings.bge_m3`, `store.faiss_store`), as the package `__init__`s are intentionally empty.
  Load `BgeM3Encoder` once and reuse it (~10 s + GPU memory).
- **ad_integration**: `core/placement.py`, `core/splitting.py` and `core/semantic.py` stay pure (no I/O).
  New position: add it to `AD_POSITIONS` in `config.py` and handle it in `place_ad()`. Splitting must not raise on
  empty text, missing punctuation or escaped whitespace (`\\n`, `\\t`).
- **ad_evaluation**: `core/masking.py::mask_ad_in_response` replaces the exact ad block with `[ad slot]` and the ad text
  is passed separately, so the judge rates *where* the ad sits, not how it reads inline. Masking is strict: if the
  rebuilt block is not in the response it raises `AdBlockNotFoundError`, and the judge services skip that row/pair,
  print a count, and do not save it (so resume retries it). There is no fuzzy fallback on purpose: the old regex
  could delete answer text when the response had its own `---`/`#`/`###` markdown. Read ad fields with `_cell`
  (NaN -> ''), matching how `place_ads` built the block. Pairwise judges run both
  orders of every pair to cancel position bias. Imports `ad_integration.core.placement` (to rebuild the ad block),
  `ad_generation.llm.parsing.extract_json_value`, and `ad_indexing.repository.JsonlAdRepository` (category grouping).
  `services.py` takes judges by injection and does not import vLLM at load time.

### Placement (`ad_integration/core/placement.py`, `semantic.py`, `splitting.py`)
Ad block is markdown: `---`, `# headline`, `### description`, `### [cta](#)`, `---`.
`first` / `last` prepend/append; `middle` splits at len//2 (paragraph, then sentence, then char fallback);
`semantic` inserts after the chunk with the highest cosine to the ad vector (`score_chunks`,
`insert_ad_after_best_paragraph`). "Paragraphs" are really 800-char LangChain chunks (CHUNK_SIZE=800),
so short responses fall through to the sentence split (pinned in `tests/test_placement.py`).

### Judges (`ad_evaluation/llm/`)
`prompts.py` (system prompts + JSON schemas), `judge.py` / `preference_judge.py` (vLLM classes on `BaseVllmGenerator`),
`parsing.py` (strict parsers). Placement comparison uses a DeepSeek-R1-Distill model with free-form reasoning then a ```json block.

### speed_comparison
Running vLLM and BGE-M3 on one GPU at once can OOM, so `run()` is phased:
1. Retrieval: `FaissVectorStore` + `BgeM3Encoder` on `--retrieval-device` (cuda default, or cpu); encode and FAISS
   search timed separately per query, sequentially; then the encoder is freed (`gc.collect`, `torch.cuda.empty_cache`).
   FAISS always runs on CPU (`faiss-cpu`). `--retrieval-only --retrieval-device cpu` needs no vLLM/GPU and writes
   `retrieval_speed_comparison_results.csv` + a summary with `llm_model="(skipped)"`.
2. LLM: `VllmBenchmarkGenerator` generates one query at a time (single-stream), checkpointing every `checkpoint_every`.
3. Summary: `compute_distribution` (mean/std/min/p50/p90/p95/p99/max), speedup = llm_ms / retrieval_ms,
   saved as `speed_comparison_summary.json`. If nothing is left to process, the summary is rebuilt from the run's CSV.

## Module dependencies
- `ad_generation`, `ad_indexing` (scripts), `ad_integration`, `ad_evaluation`, `speed_comparison` -> `common`
- `ad_integration` -> `ad_indexing` (encoder, FAISS store)
- `ad_evaluation` -> `ad_integration.core.placement`, `ad_indexing.{entities,repository}`, `ad_generation.llm.parsing`
- `speed_comparison` -> `ad_indexing`, `common`
- `ad_indexing` and `ad_generation` are coupled only through `ads.jsonl`
- Shared paths are imported from `common.shared_config` directly; module configs define only their own values
  (`ad_generation/config.py` still re-exports shared names its scripts use)

## Models and key config
| Use | Value |
|---|---|
| Ad gen + placement judge | `nvidia/Llama-3.3-70B-Instruct-FP8` (ads: len 4096, 1000 tok, gpu 0.95; judge: len 16384, 256 tok, batch 128, checkpoint 1000) |
| Categorization (legacy) | `Qwen/Qwen3-32B` (batch 32, len 32768, 3 retries, gpu 0.90) |
| Placement comparison judge | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (len 131072, 4096 tok, batch 32, fp8 hardcoded) |
| Embeddings | `BAAI/bge-m3`, `IndexFlatIP`, top-k 5 (indexing default) / 1 (integration) |
| Speed bench | 1000 queries, 512 new tokens, temp 0, shared `DEFAULT_INDEX_DIR` (`ads_index/index_new`) |
Ad generation sampling: temp 0.7, top_p 0.8, top_k 20.

## External services, env vars, hardware
- Hugging Face (dataset + model download): `HF_TOKEN` (read in `ad_generation/services.py` via dotenv)
- No DB. Everything is files. GPU required for vLLM and BGE-M3 (CPU encoder path exists only for speed benchmark)
- `common/env.py::prepend_venv_bin_to_path` edits PATH so `ninja` is found (vLLM/flashinfer JIT builds);
  the generator/judge modules call it before importing vLLM
- `datasets` (used by `prepare_dataset`) and `seaborn` (notebooks) are not in `requirements.txt` (unverified how the env gets them)

## Run, test, debug
- From repo root: `./run_vllm.sh -m src.<module>.scripts.<name> [args]` (`unset LD_LIBRARY_PATH`, runs `.lmarena-env/bin/python`; relative path, so must run from root)
- Typical order: prepare_dataset -> generate_bulk_ads -> build_index -> assign_top_ads -> place_ads -> evaluation scripts
- Most scripts accept `--limit N`; use it for smoke runs
- Tests: `./run_vllm.sh -m unittest discover -s tests -t .`. No pytest/lint/typecheck configured.
  Tests must not hit network/GPU; `test_speed_services.py` imports torch (CPU only, ~5 s)
- Debug: failures are printed (not logged); parse failures become rows (`score=0`, winner "ERROR") or are skipped

## Conventions
- Checkpointing: append-mode CSV/JSONL via `common.io.save_rows`, skip done keys on restart
- `dataclasses.asdict()` for output rows; no `to_dict()`
- Each `config.py` defines its own values directly
- Services take injected repositories/generators/encoders; scripts wire them
- LLM generators subclass `common.shared_llm.BaseVllmGenerator` (guided JSON via `json_schema`, `strip_thinking` on output)

Resume keys: assign_top_ads `id`; score_placements `(id, ad_id, position)`; preference/cross-category
`(query_id, ad_1_id, ad_2_id, is_swapped)`; placement comparison `(query_id, ad_id, pos_1, pos_2, is_swapped)`
(the last two via `common.io.load_done_keys`); speed_comparison `id` (only with explicit `--run-dir`); categories merge by `id`.

## Fragile areas and known bugs
- Masking (fixed 2026-10-29): the regex fallback could cut answer text, and placement comparison read an empty CTA as
  "nan". An audit of `ad_integration/bulk_ads/` (146,316 placed responses) found 100% exact matches and no empty CTAs,
  so existing judge results were not affected
- Parse failures are saved as completed rows (score 0 / "ERROR"), so resume never retries them
- `place_ads` has no resume and deletes outputs at start; a crash loses the run
- Resume code with missing columns or unreadable files can silently overwrite the existing output (assign_top_ads, preference services)
- `FaissVectorStore.load` (`store/faiss_store.py`) rebuilds `Ad` without `category` (sidecar has it, load ignores it).
  Low impact today: `ad_evaluation` reads categories from `ads.jsonl`, not from the loaded store
- Bulk generation appends to `ads.jsonl`; hand-editing or double-appending can produce duplicate ids, and `JsonlAdRepository.load` raises on duplicates
- speed_comparison: a default run creates a new `run_<ts>` dir so resume needs `--run-dir`; on resume the summary covers only new records; `--retrieval-only` ignores checkpoints
- assign_top_ads with no `id` column sets `id=""` for all rows; place_ads would then merge on empty ids (unverified impact)

## Where to look for X
- Shared paths and Arena constants: `common/shared_config.py`
- vLLM loading, chat template, truncation: `common/shared_llm.py`
- CSV writing, query loading, resume keys: `common/io.py`
- Prompts and JSON schemas: `ad_generation/llm/prompts.py` (category + bulk ads), `ad_evaluation/llm/prompts.py`
- LLM output parsing: `ad_generation/llm/parsing.py`, `ad_evaluation/llm/parsing.py`
- Arena filtering (English, single-turn, winner pick): `ad_generation/arena.py`, `PrepareArenaDatasetService`
- Embeddings / vector search: `ad_indexing/embeddings/bge_m3.py`, `store/faiss_store.py`, `services.py`
- Where the ad goes in a response: `ad_integration/core/placement.py`, `core/semantic.py`
- Judge scoring logic and masking: `ad_evaluation/services.py`, `core/masking.py`
- Latency stats and report: `speed_comparison/entities.py::compute_distribution`, `services.py`, `scripts/build_report.py`
- Coding standards: `.agents/rules/python.md`
