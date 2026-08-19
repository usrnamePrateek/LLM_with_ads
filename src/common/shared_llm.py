from __future__ import annotations

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.ad_generation.parsing import strip_thinking


class BaseVllmGenerator:
    """Base class for interacting with vLLM in this project."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        max_new_tokens: int,
        gpu_memory_utilization: float,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        system_prompt: str | None = None,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> None:
        if dtype == "float16" and not torch.cuda.is_available():
            raise SystemExit(f"CUDA GPU is required for {model_name} fp16")
        
        device_info = f" on {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""
        print(f"Loading {model_name} with vLLM ({dtype}){device_info} ...")

        self._llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._max_prompt_tokens = max_model_len - max_new_tokens
        self._tokenizer = self._llm.get_tokenizer()
        self._system_prompt = system_prompt
        
        structured_kwargs = {}
        if json_schema:
            structured_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=json_schema,
                disable_additional_properties=True,
            )

        self._sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            **structured_kwargs,
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
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_content})

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
