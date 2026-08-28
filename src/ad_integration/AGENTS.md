# `ad_integration` Module

Matches queries to the most relevant ad via FAISS, then injects a markdown ad block into the LLM response.

## Placement Positions

`core/placement.py` supports four strategies: `first`, `middle`, `last`, `semantic`. Adding a new position requires updating the `AD_POSITIONS` tuple in `config.py` and handling it in `place_ad()`.

## Pure Functions

`core/placement.py` and `core/splitting.py` must remain pure — no file I/O, no network, no external state. All side effects (CSV writes, FAISS queries) live in `services.py` and `repository.py`.

## Edge Case Handling

LLM responses can be empty, lack punctuation, or contain escaped whitespace (`\\n`, `\\t`). Splitting and injection logic must handle these gracefully without raising exceptions.

## Ad Assignments for Evaluation

The `AssignTopAdService` outputs a CSV of `(query, selected_ad)`. This assignment file is later consumed by the `ad_evaluation` module to perform pairwise preference testing against other ads in the same category.
