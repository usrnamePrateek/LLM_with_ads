from __future__ import annotations

from src.lmarena_prep.llm.pathutil import prepend_venv_bin_to_path

prepend_venv_bin_to_path()

from src.common.shared_llm import BaseVllmGenerator
from src.placement_testing.config import (
    JUDGE_GPU_MEMORY_UTILIZATION,
    JUDGE_MAX_MODEL_LEN,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL_NAME,
)
from src.placement_testing.prompts import PLACEMENT_JSON_SCHEMA, PLACEMENT_SYSTEM_PROMPT


class VllmPlacementJudge(BaseVllmGenerator):
    def __init__(
        self,
        model_name: str = JUDGE_MODEL_NAME,
        max_model_len: int = JUDGE_MAX_MODEL_LEN,
        gpu_memory_utilization: float = JUDGE_GPU_MEMORY_UTILIZATION,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_model_len=max_model_len,
            max_new_tokens=JUDGE_MAX_TOKENS,
            gpu_memory_utilization=gpu_memory_utilization,
            system_prompt=PLACEMENT_SYSTEM_PROMPT,
            json_schema=PLACEMENT_JSON_SCHEMA,
            temperature=0.0,
        )

