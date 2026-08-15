from __future__ import annotations

from lmarena_prep.llm.pathutil import prepend_venv_bin_to_path

prepend_venv_bin_to_path()

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from lmarena_prep.config import (
    AD_GPU_MEMORY_UTILIZATION,
    AD_MAX_MODEL_LEN,
    AD_MAX_TOKENS,
    AD_MODEL_NAME,
)
from lmarena_prep.parsing import strip_thinking
from lmarena_prep.prompts import AD_JSON_SCHEMA, build_ad_prompt


class VllmAdGenerator:
    def __init__(
        self,
        model_name: str = AD_MODEL_NAME,
        max_model_len: int = AD_MAX_MODEL_LEN,
        gpu_memory_utilization: float = AD_GPU_MEMORY_UTILIZATION,
    ) -> None:
        print(f"Loading {model_name} with vLLM ...")
        self._llm = LLM(
            model=model_name,
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._sampling = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=AD_MAX_TOKENS,
            structured_outputs=StructuredOutputsParams(json=AD_JSON_SCHEMA),
        )
        print("vLLM engine ready.")

    def generate_for_domains(self, domains: list[str]) -> list[str]:
        prompts = [build_ad_prompt(domain) for domain in domains]
        outputs = self._llm.generate(prompts, self._sampling)
        texts = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            texts.append(strip_thinking(text))
        return texts
