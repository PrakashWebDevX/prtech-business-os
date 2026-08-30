"""
LLM client helpers.

Stack is 100% free-tier:
- NVIDIA NIM (https://build.nvidia.com) for planning/writing tasks that need a
  stronger model (e.g. drafting outreach messages, research summarization).
  NIM exposes an OpenAI-compatible API, so we reuse the `openai` SDK and just
  point it at NVIDIA's base_url with an NVIDIA_API_KEY instead of OpenAI's.
- Groq for fast/cheap classification and scoring — see orchestrator/router.py,
  which already calls Groq directly.

No OpenAI usage anywhere in this project.

Model auto-discovery: NVIDIA's free NIM catalog has retired multiple models
in quick succession (meta/llama-3.1-70b-instruct, meta/llama-3.3-70b-
instruct, and nvidia/nv-embedqa-e5-v5 all went HTTP 410 Gone within the
same week during development of this project). Hardcoding a single model
string means the whole app breaks every time NVIDIA retires whatever we
picked. Instead, at first use this module calls NIM's /v1/models endpoint
(an OpenAI-compatible model listing) and picks the first live match from a
preference list below, for both the chat model and the embedding model. If
you set NVIDIA_MODEL / NVIDIA_EMBED_MODEL in .env, that's tried first and
only used if it's actually still available — otherwise auto-discovery
takes over instead of hard failing.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger("prtech.tools.llm")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Tried in order; first one that shows up in /v1/models wins. Update this
# list occasionally, but the point of auto-discovery is that the app keeps
# working even when it falls behind.
_PREFERRED_CHAT_MODELS = [
    "meta/llama-4-maverick-17b-128e-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-70b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "mistralai/mixtral-8x7b-instruct-v0.1",
]

_nim_client: OpenAI | None = None
_resolved_chat_model: str | None = None


def get_nim_client() -> OpenAI:
    global _nim_client
    if _nim_client is None:
        _nim_client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=os.environ["NVIDIA_API_KEY"],
        )
    return _nim_client


def _resolve_chat_model() -> str:
    """
    Picks a chat model that's actually live right now, caching the result
    for the life of the process. See module docstring for why this exists.
    """
    global _resolved_chat_model
    if _resolved_chat_model:
        return _resolved_chat_model

    client = get_nim_client()

    try:
        available_ids = {m.id for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "llm: could not list NIM models (%s) — falling back to NVIDIA_MODEL env var or first preferred model "
            "without verifying it's live",
            exc,
        )
        _resolved_chat_model = os.environ.get("NVIDIA_MODEL") or _PREFERRED_CHAT_MODELS[0]
        return _resolved_chat_model

    env_override = os.environ.get("NVIDIA_MODEL")
    if env_override:
        if env_override in available_ids:
            _resolved_chat_model = env_override
            logger.info("llm: using NVIDIA_MODEL override %r (confirmed live)", env_override)
            return _resolved_chat_model
        logger.warning(
            "llm: NVIDIA_MODEL=%r is set but not currently available on NIM — auto-selecting instead",
            env_override,
        )

    for candidate in _PREFERRED_CHAT_MODELS:
        if candidate in available_ids:
            _resolved_chat_model = candidate
            logger.info("llm: auto-selected NIM chat model %r", candidate)
            return _resolved_chat_model

    if available_ids:
        # Last resort: pick anything rather than hard-failing. May not be a
        # chat-capable model, but gives a clear error from the API itself
        # instead of us refusing to even try.
        _resolved_chat_model = sorted(available_ids)[0]
        logger.warning(
            "llm: none of the preferred models are available — falling back to %r. "
            "Check https://build.nvidia.com/models and update NVIDIA_MODEL in .env.",
            _resolved_chat_model,
        )
        return _resolved_chat_model

    raise RuntimeError(
        "NVIDIA NIM's /v1/models endpoint returned no models at all — check that NVIDIA_API_KEY is set and valid."
    )


def nim_complete(system_prompt: str, user_prompt: str, temperature: float = 0.4, max_tokens: int = 1024) -> str:
    """Simple single-turn completion helper for planning/writing agent nodes."""
    client = get_nim_client()
    model = _resolve_chat_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# Free NIM-hosted embedding model. 1024-dim output — schema.sql's
# `research_docs.embedding` column must match this dimension (see the
# `vector(1024)` column type there). Only models confirmed to output
# 1024-dim vectors belong in this list — swapping in a different-dimension
# model without also migrating the Supabase column will fail inserts.
_PREFERRED_EMBED_MODELS = [
    "nvidia/nv-embedqa-e5-v5",  # 1024-dim
    "baai/bge-m3",  # 1024-dim
]

_resolved_embed_model: str | None = None


def _resolve_embed_model() -> str:
    global _resolved_embed_model
    if _resolved_embed_model:
        return _resolved_embed_model

    client = get_nim_client()

    try:
        available_ids = {m.id for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "llm: could not list NIM models for embedding resolution (%s) — falling back to NVIDIA_EMBED_MODEL "
            "env var or first preferred model without verifying it's live",
            exc,
        )
        _resolved_embed_model = os.environ.get("NVIDIA_EMBED_MODEL") or _PREFERRED_EMBED_MODELS[0]
        return _resolved_embed_model

    env_override = os.environ.get("NVIDIA_EMBED_MODEL")
    if env_override:
        if env_override in available_ids:
            _resolved_embed_model = env_override
            logger.info("llm: using NVIDIA_EMBED_MODEL override %r (confirmed live)", env_override)
            return _resolved_embed_model
        logger.warning(
            "llm: NVIDIA_EMBED_MODEL=%r is set but not currently available on NIM — auto-selecting instead",
            env_override,
        )

    for candidate in _PREFERRED_EMBED_MODELS:
        if candidate in available_ids:
            _resolved_embed_model = candidate
            logger.info("llm: auto-selected NIM embedding model %r", candidate)
            return _resolved_embed_model

    raise RuntimeError(
        "None of the known 1024-dim NIM embedding models "
        f"({_PREFERRED_EMBED_MODELS}) are currently available. Check "
        "https://build.nvidia.com/models for a live embedding model, set NVIDIA_EMBED_MODEL "
        "in .env, and update schema.sql's vector(1024) column if its dimension differs."
    )


def nim_embed(text: str, input_type: str = "passage") -> list[float]:
    """
    input_type: "passage" when embedding a document to store, "query" when
    embedding a search query to look documents up — NV-EmbedQA is trained
    asymmetrically, so using the right one on each side matters for
    retrieval quality.
    """
    client = get_nim_client()
    model = _resolve_embed_model()
    resp = client.embeddings.create(
        model=model,
        input=[text],
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return resp.data[0].embedding
