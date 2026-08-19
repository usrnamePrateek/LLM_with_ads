from __future__ import annotations

from src.ad_generation.llm.pathutil import prepend_venv_bin_to_path

prepend_venv_bin_to_path()

import warnings
from src.common.shared_llm import BaseVllmGenerator
from src.ad_generation.config import (
    AD_GPU_MEMORY_UTILIZATION,
    AD_MAX_MODEL_LEN,
    AD_MAX_TOKENS,
    AD_MODEL_NAME,
)
from src.ad_generation.prompts import AD_JSON_SCHEMA, build_ad_prompt


class VllmAdGenerator(BaseVllmGenerator):
    def __init__(
        self,
        model_name: str = AD_MODEL_NAME,
        max_model_len: int = AD_MAX_MODEL_LEN,
        gpu_memory_utilization: float = AD_GPU_MEMORY_UTILIZATION,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_model_len=max_model_len,
            max_new_tokens=AD_MAX_TOKENS,
            gpu_memory_utilization=gpu_memory_utilization,
            json_schema=AD_JSON_SCHEMA,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

    def generate_for_domains(self, domains: list[str]) -> list[str]:
        prompts = [build_ad_prompt(domain) for domain in domains]
        return self.generate(prompts)
