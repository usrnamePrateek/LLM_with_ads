from __future__ import annotations

from lmarena_prep.config import CATEGORY_COLUMNS

CATEGORY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "intent": {"type": "string"},
        "commercial_intent": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": list(CATEGORY_COLUMNS),
    "additionalProperties": False,
}

AD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ads": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
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

CATEGORY_SYSTEM_PROMPT = """You classify a single user query for advertising relevance.

Return ONLY a JSON object with exactly these fields:
{"domain": "...", "intent": "...", "commercial_intent": 0}

Do not return any other fields. Do not return explanations.

Fields:
- domain: concise reusable topic label (e.g. "Sports & Fitness", "Technology", "Travel")
- intent: concise reusable user-intent label (e.g. "Information Seeking", "Product Research", "Purchase")
- commercial_intent: integer 0, 1, 2, or 3

Primary goal: USER INTENT and COMMERCIAL INTENT, not merely the topic.
Do not infer commercial intent just because the topic has products associated with it.
Informational questions about commercial topics still get commercial_intent = 0.

commercial_intent scale:
0 = No commercial intent. Pure information, explanation, creative content, general knowledge, personal advice. No obvious product/service/business opportunity.
1 = Weak commercial intent. Possible connection to a product/service, but the user is not clearly researching or considering a purchase.
2 = Moderate commercial intent. Researching, comparing, evaluating, or seeking recommendations that could reasonably lead to a purchase or paid product/service.
3 = Strong commercial intent. Explicitly wants to buy, book, order, hire, subscribe, find a provider, or take another clear commercial action.

Label style:
- Keep domain and intent concise, consistent, and reusable across a dataset.
- Prefer "Sports & Fitness" over "Sports, Exercise, Physical Activity, and Fitness".
- Prefer "Product Research" over "Researching potential products to purchase".

Examples:
Query: "What is quantum entanglement?"
{"domain": "Science", "intent": "Information Seeking", "commercial_intent": 0}

Query: "What is an iPhone?"
{"domain": "Technology", "intent": "Information Seeking", "commercial_intent": 0}

Query: "Which iPhone should I buy?"
{"domain": "Technology", "intent": "Product Research", "commercial_intent": 2}

Query: "Where can I buy an iPhone?"
{"domain": "Technology", "intent": "Purchase", "commercial_intent": 3}

Query: "What are the best running shoes for beginners?"
{"domain": "Sports & Fitness", "intent": "Product Research", "commercial_intent": 2}

Query: "Where can I buy running shoes?"
{"domain": "Sports & Fitness", "intent": "Purchase", "commercial_intent": 3}

Query: "Compare Nike and Adidas running shoes"
{"domain": "Sports & Fitness", "intent": "Product Comparison", "commercial_intent": 2}

Query: "Find a good hotel in Paris"
{"domain": "Travel", "intent": "Travel Planning", "commercial_intent": 3}

Query: "Explain how airplanes fly"
{"domain": "Science", "intent": "Information Seeking", "commercial_intent": 0}

Query: "What are some good movies to watch?"
{"domain": "Entertainment", "intent": "Recommendation", "commercial_intent": 1}

Query: "Write me a poem about the ocean"
{"domain": "Creative Writing", "intent": "Creative", "commercial_intent": 0}

Recommendations/comparisons about products/services generally get 2.
Explicit purchasing/booking/hiring/finding-provider requests generally get 3.
Creative writing and general knowledge generally get 0.
"""


def build_ad_prompt(domain: str) -> str:
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
