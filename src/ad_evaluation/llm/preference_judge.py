from __future__ import annotations

from src.ad_generation.llm.pathutil import prepend_venv_bin_to_path

prepend_venv_bin_to_path()

from src.common.shared_llm import BaseVllmGenerator
from src.ad_evaluation.config import (
    JUDGE_GPU_MEMORY_UTILIZATION,
    JUDGE_MAX_MODEL_LEN,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL_NAME,
)
from src.ad_evaluation.llm.prompts import PREFERENCE_JSON_SCHEMA, PREFERENCE_SYSTEM_PROMPT


class VllmPreferenceJudge(BaseVllmGenerator):
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
            system_prompt=PREFERENCE_SYSTEM_PROMPT,
            json_schema=PREFERENCE_JSON_SCHEMA,
            temperature=0.0,
        )
