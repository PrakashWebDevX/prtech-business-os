"""
Intent classification -> agent selection.

Uses a fast Groq-hosted model for cheap, low-latency classification rather
than the planning model, per the spec's model split (NVIDIA NIM for
planning/writing, Groq for fast classification/scoring).

Model note: llama-3.3-70b-versatile and llama-3.1-8b-instant were
deprecated by Groq on free/developer tiers as of June 17, 2026. This uses
openai/gpt-oss-20b instead — fast and free-tier friendly for a simple
classification task like this one. If you want higher accuracy on tricky
multi-part requests, swap to openai/gpt-oss-120b (Groq's recommended
replacement for the old 70B model).
"""

import json
import logging
import os

from groq import Groq

from memory.shared_state import Intent, SharedState

logger = logging.getLogger("prtech.orchestrator.router")

_INTENTS: list[Intent] = ["lead_gen", "outreach", "social", "research", "form_fill", "monitor", "clarify"]

_SYSTEM_PROMPT = f"""You are an intent classifier for a business-automation assistant.
Classify the user's message into exactly one of: {", ".join(_INTENTS)}.

- lead_gen: finding new prospects/businesses (e.g. "find dentists in Coimbatore")
- outreach: emailing/DMing leads that already exist
- social: posting content or replying to comments/DMs on social platforms
- research: answering a research question, summarizing sources
- form_fill: filling out or submitting a web form (e.g. "fill the contact form",
  "fill out this form", "submit this form with my data", "fill the test form")
- monitor: watching a URL/page for changes (e.g. "watch this page", "monitor this URL",
  "alert me if this page changes")
- clarify: the request is ambiguous, off-topic, or missing info needed to route it

Respond with ONLY a JSON object: {{"intent": "<one of the above>", "reason": "<one short sentence>"}}
No markdown, no code fences, no extra text — just the raw JSON object.
"""


def _get_groq_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Handles ```json\n{...}\n``` and plain ```\n{...}\n```
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def classify_intent(user_input: str) -> Intent:
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        max_tokens=100,
    )
    raw = _strip_markdown_fences(resp.choices[0].message.content or "")
    try:
        parsed = json.loads(raw)
        intent = parsed.get("intent")
        if intent in _INTENTS:
            return intent  # type: ignore[return-value]
        logger.warning("router: model returned an intent not in the known set: %r | raw=%r", intent, raw)
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("router: failed to parse classifier output as JSON (%s) | raw=%r", exc, raw)
    return "clarify"


def router_node(state: SharedState) -> SharedState:
    """LangGraph node: reads state['user_input'], writes state['intent']."""
    intent = classify_intent(state["user_input"])
    state["intent"] = intent
    state["active_agent"] = intent if intent != "clarify" else None
    return state
