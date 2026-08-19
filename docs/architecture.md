# Project Architecture

This document outlines the architecture, data flows, and module responsibilities for the LLM Ad Placement Testing project.

## End-to-End Data Flow

1. **Input**: User queries and human-preference LLM responses from the LMSYS Chatbot Arena dataset (`parquet` format).
2. **Data Preparation**: Queries are categorized into domains and commercial intents. Synthetic ad creatives are generated for these domains and stored as `jsonl`.
3. **Retrieval Setup**: Synthetic ads are embedded using `BAAI/bge-m3` and indexed in a FAISS vector database.
4. **Matching**: User queries are embedded, and the FAISS index retrieves the most contextually relevant ad.
5. **Generation/Placement**: The retrieved ad is formatted as Markdown and injected into the original LLM response. Placement strategies include `first`, `middle`, `last`, and `semantic` (finding the most contextually relevant paragraph boundary using cosine similarity).
6. **Evaluation**: A large LLM (Llama-3.3-70B) acting as a "judge" reviews the combined text and scores the placement (1-5) based on relevance, coherence, and disruption. Scores are output to CSV.

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
  - `VllmAdGenerator`: Generates ad copies (headlines/CTAs) (inherits `BaseVllmGenerator`).
- **Dependencies**: Depends heavily on `pandas` (for parquet manipulation) and `vLLM`.

### 3. `src/ad_indexing` (Vector Search)
- **Purpose**: Embeds ad copy and handles similarity searches.
- **Important Classes**:
  - `BgeM3Embedder`: Wraps `sentence-transformers` for embedding generation.
  - `FaissStore`: Abstraction over `faiss.IndexFlatIP` to store, save, load, and query ad embeddings.
- **Dependencies**: `faiss-cpu`, `sentence-transformers`, `numpy`.

### 4. `src/ad_integration` (Ad Placement)
- **Purpose**: Matches ads to queries and structurally injects them into the text.
- **Important Modules/Functions**:
  - `assign_top_ads.py`: Combines the parquet dataset and the FAISS index to map queries to ads.
  - `place_ads.py`: Executes the placement logic and outputs a CSV.
  - `placement.py` (`place_ad()`): Pure function that injects an ad markdown block at a specific structural point (`first`, `middle`, `last`).
  - `semantic.py` (`insert_ad_after_best_paragraph()`): Uses embeddings to find the most contextually relevant paragraph boundary for semantic placement.

### 5. `src/ad_evaluation` (Evaluation)
- **Purpose**: Uses an LLM judge to evaluate the quality of the injected ad.
- **Important Classes/Modules**:
  - `VllmPlacementJudge`: Prompts the model with a strict JSON schema for scoring. 
  - `masking.py` (`mask_ad_in_response()`): Replaces the specific ad markdown block with an `[ad slot]` placeholder to prevent the judge from being biased by the ad content itself (focusing strictly on the structural/contextual *placement*).
  - `services.py` (`PlacementTestingService`): Manages the checkpointing logic. It reads existing outputs to skip already-scored rows, preventing data loss on process interruption.

## Important Architectural Conventions
- **Repository Pattern**: CSV and FAISS storage logic are isolated behind simple repository/store classes (e.g., `PlacementScoreCsvRepository`, `FaissStore`).
- **Checkpointing**: Long-running LLM batch jobs write incrementally (append mode) to avoid catastrophic data loss on OOM or interruptions.
- **Module Independence**: Scripts are designed to run sequentially as independent steps of a pipeline, passing state via the file system (CSV/JSONL/Parquet) rather than direct memory passing.
