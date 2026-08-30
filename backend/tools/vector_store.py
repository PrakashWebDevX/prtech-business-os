"""
Thin Supabase helper shared by all agents.

Kept minimal on purpose: raw `postgrest` table calls for structured rows
(leads, outreach_log, monitor_snapshots) plus a small helper for the
research_docs pgvector table. Swap in the official `supabase-py` client's
richer query builder as needs grow.
"""

import os
from typing import Any, Optional

from supabase import Client, create_client

_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _client = create_client(url, key)
    return _client


def insert_rows(table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    resp = get_client().table(table).insert(rows).execute()
    return resp.data or []


def select_rows(
    table: str,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 100,
    order_by: Optional[str] = None,
    desc: bool = False,
) -> list[dict[str, Any]]:
    query = get_client().table(table).select("*").limit(limit)
    for col, val in (filters or {}).items():
        query = query.eq(col, val)
    if order_by:
        query = query.order(order_by, desc=desc)
    resp = query.execute()
    return resp.data or []


def upsert_rows(table: str, rows: list[dict[str, Any]], on_conflict: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    resp = get_client().table(table).upsert(rows, on_conflict=on_conflict).execute()
    return resp.data or []


def insert_research_doc(query: str, content: str, embedding: list[float], source_url: str) -> dict[str, Any]:
    resp = (
        get_client()
        .table("research_docs")
        .insert({"query": query, "content": content, "embedding": embedding, "source_url": source_url})
        .execute()
    )
    return (resp.data or [{}])[0]


def match_research_docs(embedding: list[float], match_count: int = 5) -> list[dict[str, Any]]:
    """
    Requires a `match_research_docs` RPC function defined in Supabase, e.g.:

    create or replace function match_research_docs(query_embedding vector(1024), match_count int)
    returns setof research_docs language sql as $$
      select * from research_docs
      order by embedding <-> query_embedding
      limit match_count;
    $$;
    """
    resp = get_client().rpc(
        "match_research_docs",
        {"query_embedding": embedding, "match_count": match_count},
    ).execute()
    return resp.data or []
