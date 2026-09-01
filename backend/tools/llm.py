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


# Free NIM-hosted embedding model. schema.sql's `research_docs.embedding`
# column is `vector(2048)`, matching nvidia/nemotron-3-embed-1b's actual
# output dimension (confirmed at runtime — NVIDIA doesn't document this
# clearly, and it's not the same as the 1024-dim nv-embedqa-e5-v5 this
# project started with, which was later retired). nim_embed() checks the
# actual output dimension at runtime and raises a clear, actionable error
# (with the exact SQL to fix it) if a fallback model's dimension doesn't
# match the schema, rather than letting Supabase's insert fail with a
# confusing generic error.
_PREFERRED_EMBED_MODELS = [
    "nvidia/nemotron-3-embed-1b",  # 2048-dim — confirmed working as of this fix
    "nvidia/nv-embedqa-e5-v5",  # 1024-dim if it ever comes back — would need a schema migration to use
    "baai/bge-m3",  # 1024-dim if it ever comes back — would need a schema migration to use
    "nvidia/nv-embed-v1",  # dimension not yet confirmed by us — verified at runtime
]

_EXPECTED_EMBED_DIM = 2048  # must match schema.sql's vector(2048) column

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

    if available_ids:
        # Last resort: pick anything embedding-model-shaped rather than
        # hard-failing outright. If it's not actually an embedding model,
        # or the dimension doesn't match, nim_embed's dimension check below
        # will catch it with a clear error.
        for model_id in sorted(available_ids):
            if "embed" in model_id.lower():
                _resolved_embed_model = model_id
                logger.warning(
                    "llm: none of the preferred embedding models are available — falling back to %r "
                    "(unverified dimension). Check https://build.nvidia.com/models and update "
                    "NVIDIA_EMBED_MODEL in .env if this doesn't work.",
                    model_id,
                )
                return _resolved_embed_model

    raise RuntimeError(
        "No embedding-capable NIM model could be found at all (checked preferred models "
        f"{_PREFERRED_EMBED_MODELS} and scanned for any model with 'embed' in its name). "
        "Check https://build.nvidia.com/models for a live embedding model and set "
        "NVIDIA_EMBED_MODEL in .env."
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
    embedding = resp.data[0].embedding

    actual_dim = len(embedding)
    if actual_dim != _EXPECTED_EMBED_DIM:
        raise ValueError(
            f"Embedding model {model!r} returned a {actual_dim}-dim vector, but Supabase's "
            f"research_docs.embedding column is vector({_EXPECTED_EMBED_DIM}). Fix by running this "
            "in the Supabase SQL Editor (safe if the table is currently empty; if it has rows, "
            "back them up first since this drops existing embeddings):\n"
            f"  alter table research_docs alter column embedding type vector({actual_dim});\n"
            f"  create or replace function match_research_docs(query_embedding vector({actual_dim}), "
            "match_count int) returns setof research_docs language sql as $$ select * from "
            "research_docs order by embedding <-> query_embedding limit match_count; $$;\n"
            "Then also update schema.sql to match so future setups don't hit this again."
        )

    return embedding
