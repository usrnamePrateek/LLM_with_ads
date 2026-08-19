from __future__ import annotations

import re

from src.ad_integration.core.placement import format_ad_block
from src.ad_evaluation.config import AD_SLOT

_AD_BLOCK_RE = re.compile(
    r"---\s*\n+# [^\n]+\n+### .+\n+### \[[^\]]+\]\(#\)\s*\n+---",
    re.DOTALL,
)


def mask_ad_in_response(
    response_with_ad: str,
    headline: str,
    description: str,
    cta: str,
) -> tuple[str, str]:
    """Return (masked_response, ad_text)."""
    ad_text = format_ad_block(headline, description, cta)
    body = response_with_ad or ""
    if ad_text in body:
        return body.replace(ad_text, AD_SLOT, 1), ad_text
    match = _AD_BLOCK_RE.search(body)
    if match:
        return body[: match.start()] + AD_SLOT + body[match.end() :], match.group(0)
    return body, ad_text
