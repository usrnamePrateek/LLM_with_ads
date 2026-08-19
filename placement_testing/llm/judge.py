from __future__ import annotations

from lmarena_prep.llm.pathutil import prepend_venv_bin_to_path

prepend_venv_bin_to_path()

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from lmarena_prep.parsing import strip_thinking
from placement_testing.config import (
    JUDGE_GPU_MEMORY_UTILIZATION,
    JUDGE_MAX_MODEL_LEN,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL_NAME,
)
from placement_testing.prompts import PLACEMENT_JSON_SCHEMA, PLACEMENT_SYSTEM_PROMPT


class VllmPlacementJudge:
    def __init__(
        self,
        model_name: str = JUDGE_MODEL_NAME,
        max_model_len: int = JUDGE_MAX_MODEL_LEN,
        gpu_memory_utilization: float = JUDGE_GPU_MEMORY_UTILIZATION,
    ) -> None:
        print(f"Loading {model_name} with vLLM ...")
        self._llm = LLM(
            model=model_name,
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._max_prompt_tokens = max_model_len - JUDGE_MAX_TOKENS
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling = SamplingParams(
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
            structured_outputs=StructuredOutputsParams(
                json=PLACEMENT_JSON_SCHEMA,
                disable_additional_properties=True,
            ),
        )
        print(
            "vLLM engine ready. "
            f"max_model_len={max_model_len} max_prompt_tokens={self._max_prompt_tokens}"
        )

    def generate(self, user_prompts: list[str]) -> list[str]:
        formatted = [self._format_prompt(prompt) for prompt in user_prompts]
        outputs = self._llm.generate(formatted, self._sampling)
        texts = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            texts.append(strip_thinking(text))
        return texts

    def _apply_chat_template(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": PLACEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _truncate_user(self, user_content: str) -> str:
        prompt = self._apply_chat_template(user_content)
        prompt_ids = self._encode(prompt)
        overflow = len(prompt_ids) - self._max_prompt_tokens
        if overflow <= 0:
            return user_content

        user_ids = self._encode(user_content)
        keep = max(0, len(user_ids) - overflow)
        truncated = self._tokenizer.decode(user_ids[:keep], skip_special_tokens=True)
        while keep > 0:
            prompt_ids = self._encode(self._apply_chat_template(truncated))
            if len(prompt_ids) <= self._max_prompt_tokens:
                break
            keep = max(0, keep - (len(prompt_ids) - self._max_prompt_tokens))
            truncated = self._tokenizer.decode(user_ids[:keep], skip_special_tokens=True)
        print(
            f"  truncated user prompt from {len(user_ids)} to {keep} tokens"
        )
        return truncated

    def _format_prompt(self, user_content: str) -> str:
        return self._apply_chat_template(self._truncate_user(user_content))
