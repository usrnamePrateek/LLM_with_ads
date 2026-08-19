from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingEncoder(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        ...

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        ...
