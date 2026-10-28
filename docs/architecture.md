# Project Architecture

This document outlines the architecture, data flows, and module responsibilities for the LLM Ad Placement Testing project.

## End-to-End Data Flow

1. **Input**: User queries and human-preference LLM responses from the LMSYS Chatbot Arena dataset (`parquet` format).
2. **Data Preparation**: Queries are categorized into domains and commercial intents. Synthetic ad creatives are generated per taxonomy sub-topic (`taxonomy_subtopics.csv`) in bulk and appended to `ads.jsonl`.
3. **Retrieval Setup**: Synthetic ads are embedded using `BAAI/bge-m3` and indexed in a FAISS vector database.
4. **Matching**: User queries are embedded, and the FAISS index retrieves the most contextually relevant ad.
5. **Generation/Placement**: The retrieved ad is formatted as Markdown and injected into the original LLM response. Placement strategies include `first`, `middle`, `last`, and `semantic` (finding the most contextually relevant paragraph boundary using cosine similarity).
6. **Evaluation**: A large LLM (Llama-3.3-70B) acting as a "judge" reviews the combined text and scores the placement (1-5) based on relevance, coherence, and disruption. Scores are output to CSV.

## Data locations
Pipeline outputs live under `data/processed/generation/bulk_ads/` (`query_top1_ads.csv`, `query_ad_positions.csv`, `query_semantic_paragraphs.csv`); ads and the FAISS index under `data/processed/ads/`; evaluation outputs under `data/evaluation/` and `data/processed/ad_evaluation/`.

## External Services & Models
- **Local Inference**: `vLLM` is used to host all Llama and Qwen models.
- **Embeddings**: `sentence-transformers` runs the `BAAI/bge-m3` model for vector embeddings.
- **Vector Store**: `faiss-cpu` handles the similarity search.

---

## Modules

### 1. `src/common` (Shared Infrastructure)
- **Purpose**: Provides centralized configuration and base classes for the entire project.
- **Important Classes/Files**:
  - `shared_config.py`: Contains global file paths (`REPO_ROOT`, `DEFAULT_OUTPUT_CSV`), model names, and batch sizes.
  - `shared_llm.py` (`BaseVllmGenerator`): Base class that encapsulates `vLLM` instantiation, memory utilization, chat templates, and token truncation. Inherited by all task-specific LLM generators to prevent duplicate boilerplates.

### 2. `src/ad_generation` (Data Preparation)
- **Purpose**: Downloads the Arena dataset, categorizes queries, and generates ads.
- **Important Classes**:
  - `VllmCategoryGenerator`: Evaluates query domain/intent (inherits `BaseVllmGenerator`).
  - `VllmBulkAdGenerator`: Generates several ad copies (headline/description/CTA) per taxonomy sub-topic (inherits `BaseVllmGenerator`). Entry point: `generate_bulk_ads.py`; resumes from per-domain counts already in `ads.jsonl`.
- **Dependencies**: Depends heavily on `pandas` (for parquet manipulation) and `vLLM`.

### 3. `src/ad_indexing` (Vector Search)
- **Purpose**: Embeds ad copy and handles similarity searches.
- **Important Classes**:
  - `BgeM3Encoder` (`embeddings/bge_m3.py`): Wraps `sentence-transformers` for embedding generation.
  - `FaissVectorStore` (`store/faiss_store.py`, implements the `VectorStore` protocol): Abstraction over `faiss.IndexFlatIP` to store, save, load, and query ad embeddings.
- **Dependencies**: `faiss-cpu`, `sentence-transformers`, `numpy`.

### 4. `src/ad_integration` (Ad Placement)
- **Purpose**: Matches ads to queries and structurally injects them into the text.
- **Important Modules/Functions**:
  - `assign_top_ads.py`: Combines the parquet dataset and the FAISS index to map queries to ads.
  - `place_ads.py`: Executes the placement logic and outputs a CSV.
  - `core/placement.py` (`place_ad()`): Pure function that injects an ad markdown block at a specific position (`first`, `middle`, `last`, `semantic`).
  - `core/semantic.py` (`insert_ad_after_best_paragraph()`): Uses embeddings to find the most contextually relevant paragraph boundary for semantic placement.

### 5. `src/ad_evaluation` (Evaluation)
- **Purpose**: Uses an LLM judge to evaluate the quality of the injected ad.
- **Important Classes/Modules**:
  - `VllmPlacementJudge`: Prompts the model with a strict JSON schema for scoring. 
  - `masking.py` (`mask_ad_in_response()`): Replaces the specific ad markdown block with an `[ad slot]` placeholder to prevent the judge from being biased by the ad content itself (focusing strictly on the structural/contextual *placement*).
  - `services.py` (`ScorePlacementsService`): Manages the checkpointing logic. It reads existing outputs to skip already-scored rows, preventing data loss on process interruption.
  - Pairwise evaluations (both orders per pair, to remove position bias): `run_preference_eval.py` (ad vs ads in the same category), `run_cross_category_eval.py` (ad vs ads in other categories), `run_placement_comparison.py` (`semantic` vs `first`/`middle`/`last`, judged by DeepSeek-R1-Distill-Qwen-32B).

### 6. `src/speed_comparison` (Latency Benchmark)
- **Purpose**: Compares query-time ad retrieval latency (BGE-M3 encode + FAISS search) with LLM response generation latency.
- **Important Classes**: `SpeedComparisonOrchestrator` (phased run: retrieval, then LLM, then summary), `RetrievalBenchmarkService`, `VllmBenchmarkGenerator`, `compute_distribution()` in `entities.py`. See `src/speed_comparison/AGENTS.md`.
- `--batch-size` is accepted but reserved for future batching; runs are currently single-stream.

## Important Architectural Conventions
- **Repository Pattern**: CSV and FAISS storage logic are isolated behind simple repository/store classes (e.g., `PlacementScoreCsvRepository`, `FaissVectorStore`).
- **Checkpointing**: Long-running LLM batch jobs write incrementally (append mode) to avoid catastrophic data loss on OOM or interruptions.
- **Module Independence**: Scripts are designed to run sequentially as independent steps of a pipeline, passing state via the file system (CSV/JSONL/Parquet) rather than direct memory passing.
