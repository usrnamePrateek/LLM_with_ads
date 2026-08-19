from __future__ import annotations

from lmarena_prep.llm.pathutil import prepend_venv_bin_to_path

prepend_venv_bin_to_path()

from shared_llm import BaseVllmGenerator
from lmarena_prep.config import (
    CATEGORY_GPU_MEMORY_UTILIZATION,
    CATEGORY_MAX_MODEL_LEN,
    CATEGORY_MAX_NEW_TOKENS,
    CATEGORY_MODEL_NAME,
)
from lmarena_prep.prompts import CATEGORY_JSON_SCHEMA, CATEGORY_SYSTEM_PROMPT


class VllmCategoryGenerator(BaseVllmGenerator):
    def __init__(
        self,
        model_name: str = CATEGORY_MODEL_NAME,
        max_model_len: int = CATEGORY_MAX_MODEL_LEN,
        gpu_memory_utilization: float = CATEGORY_GPU_MEMORY_UTILIZATION,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_model_len=max_model_len,
            max_new_tokens=CATEGORY_MAX_NEW_TOKENS,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            system_prompt=CATEGORY_SYSTEM_PROMPT,
            json_schema=CATEGORY_JSON_SCHEMA,
            temperature=0.0,
        )

    def generate(self, queries: list[str]) -> list[str]:
        prompts = [f"Query: {q}" for q in queries]
        return super().generate(prompts)

