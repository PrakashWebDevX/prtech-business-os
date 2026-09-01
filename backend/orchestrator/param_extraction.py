"""
LLM-based param extraction from free-text chat messages.

Replaces the naive `" in "` string-split that lead_gen and outreach nodes
used to route free text into structured params — that broke on almost any
phrasing that wasn't exactly "<niche> in <location>" (e.g. "find me some
dentists near Chennai", "look for plumbers around Bangalore", "any
electricians close to Pune?").

Uses NVIDIA NIM (already wired up in tools/llm.py) rather than a second
LLM provider, since this is a planning/extraction task, not the fast
classification task orchestrator/router.py uses Groq for.

Falls back to a naive split if the LLM call fails or returns unparseable
output, so a transient NIM issue degrades gracefully instead of breaking
the whole request.
"""

import json
import logging

from tools.llm import nim_complete

logger = logging.getLogger("prtech.orchestrator.param_extraction")

_NICHE_LOCATION_SYSTEM_PROMPT = """Extract a business niche/category and a location from the
user's message. The message might be phrased many different ways — handle all of them:
"find dentists in Coimbatore", "look for some plumbers near Chennai", "any electricians
around Bangalore?", "get me a list of cafes close to downtown Austin", etc.

Rules:
- niche: the type of business/profession, singular or plural is fine, lowercase.
- location: the place name as given (city, neighborhood, area — whatever the user said).
  If no location is mentioned at all, use null.
- If no niche is mentioned at all, use null.

Respond with ONLY a JSON object: {"niche": "<string or null>", "location": "<string or null>"}
No markdown, no code fences, no extra text — just the raw JSON object.
"""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _naive_fallback(user_input: str) -> dict:
    niche, _, location = user_input.partition(" in ")
    niche = niche.replace("find", "").replace("Find", "").strip() or None
    location = location.strip() or None
    return {"niche": niche, "location": location}


def extract_niche_location(user_input: str) -> dict:
    """
    Returns {"niche": str | None, "location": str | None}.
    Used by both lead_gen and outreach nodes — both need the same
    niche/location extraction from a free-text message.
    """
    try:
        raw = nim_complete(_NICHE_LOCATION_SYSTEM_PROMPT, user_input, temperature=0, max_tokens=100)
        cleaned = _strip_markdown_fences(raw)
        parsed = json.loads(cleaned)
        niche = parsed.get("niche") or None
        location = parsed.get("location") or None
        return {"niche": niche, "location": location}
    except Exception as exc:  # noqa: BLE001 - degrade to naive parsing rather than fail the request
        logger.warning(
            "param_extraction: LLM extraction failed (%s) — falling back to naive string split for %r",
            exc,
            user_input,
        )
        return _naive_fallback(user_input)
