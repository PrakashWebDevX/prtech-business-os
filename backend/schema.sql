-- Run this in the Supabase SQL editor before starting the backend.
create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists leads (
  id uuid primary key default gen_random_uuid(),
  niche text,
  location text,
  business_name text,
  phone text,
  email text,
  website text,
  socials jsonb,
  score float,
  created_at timestamptz default now()
);

create table if not exists outreach_log (
  id uuid primary key default gen_random_uuid(),
  lead_id uuid references leads(id),
  channel text,
  message text,
  status text,
  sent_at timestamptz
);

create table if not exists research_docs (
  id uuid primary key default gen_random_uuid(),
  query text,
  content text,
  -- dimension for NVIDIA NIM's nvidia/nemotron-3-embed-1b (free tier).
  -- NVIDIA has retired embedding models before (nv-embedqa-e5-v5 went 410
  -- Gone); if tools/llm.py's auto-discovery falls back to a different
  -- model with a different dimension, nim_embed() will raise an error
  -- containing the exact ALTER TABLE / CREATE FUNCTION SQL to run here.
  embedding vector(2048),
  source_url text,
  created_at timestamptz default now()
);

create table if not exists monitor_snapshots (
  id uuid primary key default gen_random_uuid(),
  url text,
  content_hash text,
  captured_at timestamptz default now()
);

create table if not exists agent_audit_log (
  id uuid primary key default gen_random_uuid(),
  agent text,
  action text,
  input jsonb,
  output jsonb,
  success boolean,
  created_at timestamptz default now()
);

-- Optional: similarity search RPC used by tools/vector_store.match_research_docs
create or replace function match_research_docs(query_embedding vector(2048), match_count int)
returns setof research_docs
language sql
as $$
  select * from research_docs
  order by embedding <-> query_embedding
  limit match_count;
$$;
