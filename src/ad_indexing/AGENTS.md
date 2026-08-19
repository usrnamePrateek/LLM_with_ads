# Agent Instructions: `ad_indexing` Module

## Purpose
This module handles semantic similarity and vector search. It is responsible for embedding textual data (ad creatives and user queries) and retrieving the most relevant ads using a vector database.

## Responsibilities
- Generate high-quality embeddings for text.
- Build, persist, and query vector indices.
- **Do not** put business logic related to ad placement or data cleaning here. This module is strictly for vector mathematics and storage operations.

## Dependencies
- **Allowed**: `src.common` (for configuration).
- **External**: `sentence-transformers` (BAAI/bge-m3), `faiss-cpu`, `numpy`.
- **Forbidden**: Do not import from `ad_integration`, `ad_generation`, or `ad_evaluation` to maintain a one-way dependency flow.

## Important Interfaces
- `BgeM3Embedder`: Wrapper for the embedding model.
- `FaissStore`: Abstraction over the `faiss.IndexFlatIP` index.
- `Store` (Base Interface): Defines the contract for any vector store implementation.

## Design Rules
- **Dependency Inversion / Adapter Pattern**: Keep the specific vector database implementation (FAISS) isolated behind the `Store` interface. This allows us to easily swap FAISS for Pinecone, Milvus, or another DB in the future.
- **Single Responsibility Principle**: Ensure embedding logic and storage logic remain in separate classes.

## Data Flow
1. Receives raw JSONL ad data or query strings.
2. Converts text to dense vectors via `BgeM3Embedder`.
3. Ingests vectors into `FaissStore`.
4. Returns top-K nearest neighbors and cosine similarity scores.

## Testing
- Mock the `BgeM3Embedder` to return static numpy arrays during unit tests to avoid loading heavy torch models in CI.
- Test the `Store` interface implementations directly with synthetic vectors.

## Performance Considerations
- **Vectorization**: Always pass lists of strings to the embedder (batching) rather than embedding one string at a time in a for-loop.
- **Model Reuse**: Load the embedding model into memory once and reuse it across the module's lifecycle. Do not repeatedly instantiate the model.
