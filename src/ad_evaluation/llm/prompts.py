"""Prompts and schemas for the LLM-as-a-judge placement evaluator."""
from __future__ import annotations

PLACEMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

PLACEMENT_SYSTEM_PROMPT = """You are a digital marketing expert. Your task is to evaluate how naturally and appropriately an advertisement fits at a specific position within an LLM-generated response.

Evaluate ONLY the suitability of the ad placement. Do not judge whether the advertisement itself is good, whether the product is good, or whether the user should buy it.

Consider:

1. Contextual relevance:
   Is the advertisement meaningfully related to the surrounding content and the user's intent?

2. Contextual coherence:
   Does inserting the advertisement at this position feel natural given the text immediately before and after it?

3. Flow disruption:
   Does the advertisement interrupt an important explanation, argument, list, instruction, or narrative?

4. User intent alignment:
   Would this advertisement plausibly be useful to the user given their query?

5. Placement appropriateness:
   Is this a particularly suitable point in the response to show this advertisement?

Give a score from 1 to 5:

1 = Very poor placement
2 = Poor placement
3 = Moderately appropriate
4 = Good placement
5 = Excellent placement

Return ONLY valid JSON in the following format:

{
  "score": <1-5>,
  "reason": "<brief explanation>"
}

Evaluate the suitability of inserting this advertisement at this position.
"""

PREFERENCE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "winner_ad": {"type": "string", "enum": ["ad_1", "ad_2"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["winner_ad", "confidence"],
    "additionalProperties": False,
}

PREFERENCE_SYSTEM_PROMPT = """You are a digital marketing expert evaluating the relevance of advertisements for a specific user query.

Your task is to review two advertisements (Ad 1 and Ad 2) and determine which one is more relevant, useful, and appropriate for a given user query.

Consider:
1. Keyword and intent alignment with the query.
2. Value proposition clarity.
3. Call-to-action relevance.

You must output ONLY valid JSON in the following format:
{
  "winner_ad": "ad_1" or "ad_2",
  "confidence": "low", "medium", or "high"
}
"""


def build_user_prompt(query: str, masked_response: str, ad_text: str) -> str:
    """Formats the user query, the masked LLM response, and the ad text for evaluation."""
    return (
        f"Query: {query}\n\n"
        f"Response with ad slot:\n{masked_response}\n\n"
        f"ad:\n{ad_text}"
    )


def build_pairwise_preference_prompt(query: str, ad_1_text: str, ad_2_text: str) -> str:
    """Formats the user query and two ads for preference comparison."""
    return (
        f"Query: {query}\n\n"
        f"--- Ad 1 ---\n{ad_1_text}\n\n"
        f"--- Ad 2 ---\n{ad_2_text}\n"
    )
