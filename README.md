# LLM Ad Placement Testing

This repository contains a data pipeline and evaluation framework for testing the placement of advertisements within LLM-generated responses. It uses the LMSYS Chatbot Arena dataset as a source of queries and responses, generates synthetic advertisements, places them into the responses, and evaluates the naturalness of the placements using a strong LLM as a judge.

## Project Structure

The project is broken down into four main modules, executed sequentially:

### 1. `ad_generation/` (Data Preparation)
Prepares the base dataset and generates the synthetic advertisements.
- **`scripts/prepare_dataset.py`**: Downloads and formats the LMArena human preference dataset.
- **`scripts/generate_categories.py`**: Uses an LLM to categorize queries by domain, intent, and commercial intent.
- **`scripts/generate_ads.py`**: Generates synthetic ad creatives (headlines, descriptions, CTAs) tailored to the discovered domains.

### 2. `ad_indexing/` (Vector Search)
Indexes the generated advertisements for semantic retrieval.
- **`scripts/build_index.py`**: Embeds the generated ad catalog into a FAISS index using `BAAI/bge-m3`.
- **`scripts/query.py`**: Utility for testing the retrieval index.

### 3. `ad_integration/` (Ad Placement)
Matches user queries to relevant ads and inserts them into the generated responses.
- **`scripts/assign_top_ads.py`**: Uses the FAISS index to assign the most contextually relevant ad to each query.
- **`scripts/place_ads.py`**: Injects the assigned ad into the LLM response at various structural positions (e.g., `first`, `middle`, `last`, `semantic`).

### 4. `ad_evaluation/` (Evaluation)
Evaluates the quality and flow of the injected advertisements.
- **`scripts/score_placements.py`**: Uses a large LLM (e.g., Llama-3.3-70B) acting as an expert digital marketer to score the contextual relevance, coherence, and flow disruption of the ad placement on a 1-5 scale. Supports checkpointing/resumption.

## Shared Configuration (`src/common/`)

- **`shared_config.py`**: Contains global configuration variables (model names, file paths, batch sizes) used across all modules.
- **`shared_llm.py`**: Contains the `BaseVllmGenerator` class, which handles centralized `vLLM` instantiation, token truncation, and chat template formatting.

## Usage

All scripts are designed to be run as modules from the project root using the `.lmarena-env` virtual environment. 

Example:
```bash
# Run the placement evaluation judge
python -m src.ad_evaluation.scripts.score_placements --limit 100
```

