"""LLM prompts and JSON schemas used for bulk ad generation."""
from __future__ import annotations

BULK_AD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ads": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "description": {"type": "string"},
                    "cta": {"type": "string"},
                },
                "required": ["headline", "description", "cta"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ads"],
    "additionalProperties": False,
}

def build_bulk_ad_prompt(domain: str, num_ads: int) -> str:
    return f"""
Act as a world-class, highly creative copywriter. Your task is to generate exactly {num_ads} distinct textual advertisements for the following domain:

Domain: {domain}

Requirements:
- The advertisements must use different advertising angles (e.g., emotional appeal, problem-solution, urgency, benefit-driven).
- Keep the tone concise, natural, and highly engaging.
- Make them persuasive and compelling without sounding overly promotional or spammy.
- Feel free to invent realistic fictional brands, prices, discounts, statistics,
  guarantees, or product features to make the ads feel highly authentic.

Generate only the requested advertisement data in the specified JSON format.
"""
