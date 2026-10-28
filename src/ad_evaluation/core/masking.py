"""Utilities for masking ad text from LLM judge prompts."""
from __future__ import annotations

from src.ad_integration.core.placement import format_ad_block
from src.ad_evaluation.config import AD_SLOT


class AdBlockNotFoundError(ValueError):
    """The rebuilt ad block does not occur in the response, so it cannot be masked safely."""


def mask_ad_in_response(
    response_with_ad: str,
    headline: str,
    description: str,
    cta: str,
) -> tuple[str, str]:
    """Replaces the exact ad block in the response with a generic slot token.

    There is deliberately no fuzzy fallback: guessing where the ad is can cut answer text or leave
    the ad visible to the judge. Callers skip responses that raise AdBlockNotFoundError.

    Returns:
        A tuple of (masked_response, original_ad_text).
    """
    ad_text = format_ad_block(headline, description, cta)
    body = response_with_ad or ""
    if ad_text not in body:
        raise AdBlockNotFoundError("ad block not found in response")
    return body.replace(ad_text, AD_SLOT, 1), ad_text
