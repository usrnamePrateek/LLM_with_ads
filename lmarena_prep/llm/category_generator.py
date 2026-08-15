from __future__ import annotations

from lmarena_prep.llm.pathutil import prepend_venv_bin_to_path

prepend_venv_bin_to_path()

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from lmarena_prep.config import (
    CATEGORY_GPU_MEMORY_UTILIZATION,
    CATEGORY_MAX_MODEL_LEN,
    CATEGORY_MAX_NEW_TOKENS,
    CATEGORY_MODEL_NAME,
)
from lmarena_prep.parsing import strip_thinking
from lmarena_prep.prompts import CATEGORY_JSON_SCHEMA, CATEGORY_SYSTEM_PROMPT


class VllmCategoryGenerator:
    def __init__(
        self,
        model_name: str = CATEGORY_MODEL_NAME,
        max_model_len: int = CATEGORY_MAX_MODEL_LEN,
        gpu_memory_utilization: float = CATEGORY_GPU_MEMORY_UTILIZATION,
    ) -> None:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU is required for Qwen3-32B fp16")
        print(f"Loading {model_name} with vLLM fp16 on {torch.cuda.get_device_name(0)} ...")
        self._llm = LLM(
            model=model_name,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
        )
        self._max_prompt_tokens = max_model_len - CATEGORY_MAX_NEW_TOKENS
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling = SamplingParams(
            temperature=0.0,
            max_tokens=CATEGORY_MAX_NEW_TOKENS,
            structured_outputs=StructuredOutputsParams(
                json=CATEGORY_JSON_SCHEMA,
                disable_additional_properties=True,
            ),
        )
        print(
            "vLLM engine ready. "
            f"max_model_len={max_model_len} max_prompt_tokens={self._max_prompt_tokens}"
        )

    def generate(self, prompts: list[str]) -> list[str]:
        formatted = [self._format_prompt(query) for query in prompts]
        outputs = self._llm.generate(formatted, self._sampling)
        texts = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            texts.append(strip_thinking(text))
        return texts

    def _apply_chat_template(self, query: str) -> str:
        messages = [
            {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _truncate_query(self, query: str) -> str:
        prompt = self._apply_chat_template(query)
        prompt_ids = self._encode(prompt)
        overflow = len(prompt_ids) - self._max_prompt_tokens
        if overflow <= 0:
            return query

        query_ids = self._encode(query)
        keep = max(0, len(query_ids) - overflow)
        truncated = self._tokenizer.decode(query_ids[:keep], skip_special_tokens=True)

        while keep > 0:
            prompt_ids = self._encode(self._apply_chat_template(truncated))
            if len(prompt_ids) <= self._max_prompt_tokens:
                break
            keep = max(0, keep - (len(prompt_ids) - self._max_prompt_tokens))
            truncated = self._tokenizer.decode(query_ids[:keep], skip_special_tokens=True)

        print(
            f"  truncated query from {len(query_ids)} to {keep} tokens "
            f"(prompt {len(self._encode(prompt))} -> "
            f"{len(self._encode(self._apply_chat_template(truncated)))})"
        )
        return truncated

    def _format_prompt(self, query: str) -> str:
        return self._apply_chat_template(self._truncate_query(query))
