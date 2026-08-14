import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


# =========================
# Configuration
# =========================

MODEL = "nvidia/Llama-3.3-70B-Instruct-FP8"
INPUT_FILE = "/home/gpuuser7/gpuuser7_a/prateek/LLM_with_ads/data/processed/lmarena/domains.txt"
OUTPUT_FILE = "ads2.jsonl"

MAX_MODEL_LEN = 4096
MAX_TOKENS = 1000
GPU_MEMORY_UTILIZATION = 0.90


# =========================
# Load model
# =========================

llm = LLM(
    model=MODEL,
    dtype="auto",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)


# =========================
# JSON schema
# =========================

ad_schema = {
    "type": "object",
    "properties": {
        "ads": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "headline": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    },
                    "cta": {
                        "type": "string"
                    }
                },
                "required": [
                    "headline",
                    "description",
                    "cta"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["ads"],
    "additionalProperties": False
}


structured_outputs = StructuredOutputsParams(
    json=ad_schema
)


# =========================
# Sampling parameters
# =========================

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=MAX_TOKENS,
    structured_outputs=structured_outputs,
)


# =========================
# Prompt
# =========================

def build_prompt(domain: str) -> str:
    return f"""
Generate exactly 2 textual advertisements for this domain:

Domain: {domain}

Requirements:
- The two advertisements must use different advertising angles.
- Keep them concise and natural.
- Make them persuasive but not spammy.
- Do not invent specific brands, prices, discounts, statistics,
  guarantees, or product features.
- Advertisement 1 should use a direct, benefit-focused angle.
- Advertisement 2 should use a different angle such as lifestyle,
  convenience, problem-solution, urgency, trust, or emotional appeal.

Generate only the requested advertisement data.
"""


# =========================
# Read domains
# =========================

domains = [
    line.strip()
    for line in Path(INPUT_FILE).read_text(encoding="utf-8").splitlines()
    if line.strip()
]

prompts = [
    build_prompt(domain)
    for domain in domains
]


# =========================
# Generate
# =========================

outputs = llm.generate(
    prompts,
    sampling_params=sampling_params,
)


# =========================
# Save
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for domain, output in zip(domains, outputs):

        text = output.outputs[0].text.strip()

        try:
            result = json.loads(text)

            for ad_id, ad in enumerate(result["ads"], start=1):

                record = {
                    "domain": domain,
                    "ad_id": ad_id,
                    "headline": ad["headline"],
                    "description": ad["description"],
                    "cta": ad["cta"],
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

        except Exception as e:
            print(f"Failed for {domain}: {e}")
            print(f"Raw output: {text}")


print(f"Saved {len(domains) * 2} ads to {OUTPUT_FILE}")