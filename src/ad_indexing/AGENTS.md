# `ad_indexing` Module

Embeds ad text and handles FAISS vector search. No ad placement or business logic belongs here.

## `VectorStore` Protocol

New vector store implementations (e.g. Pinecone, Milvus) must implement the `VectorStore` protocol in `store/base.py`. Do not access FAISS directly from other modules — use the store abstraction.

## Embedding Model

Uses `BAAI/bge-m3` via `sentence-transformers`. The embedder must be instantiated once and reused — loading the model is expensive (~10s + GPU memory).
