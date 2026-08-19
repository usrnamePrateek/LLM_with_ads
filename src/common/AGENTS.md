# Agent Instructions: `common` Module

## Purpose
This module serves as the shared infrastructure layer for the entire project. It contains global configuration constants and base classes that are shared across multiple high-level modules.

## Responsibilities
- Provide centralized configuration variables (e.g., file paths, global model names, batch sizes).
- Provide foundational classes used by other modules (e.g., `BaseVllmGenerator`).
- **Do not** put application-specific business logic here.

## Dependencies
- **Forbidden**: This module is the foundation of the repository. It **MUST NOT** import from `src.ad_indexing`, `src.ad_integration`, `src.ad_generation`, or `src.ad_evaluation`. Doing so will create circular import errors and violate the architectural hierarchy.

## Important Interfaces
- `shared_config.py`: Single source of truth for global configuration.
- `shared_llm.py`: Contains `BaseVllmGenerator`, which manages `vLLM` instantiation, memory constraints, chat template logic, and token truncation.

## Design Rules
- Keep this module as thin and minimal as possible.
- If a utility function or configuration is only used by one module, it belongs in that module's `config.py` or `utils.py`, not here.

## Testing
- Ensure that modifications to `BaseVllmGenerator` do not break the token truncation logic for any of the inherited classes. Tests for this module should focus purely on text truncation and tokenization behavior.
