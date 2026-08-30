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
  embedding vector(1024), -- dimension for NVIDIA NIM's nvidia/nv-embedqa-e5-v5 (free tier)
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
create or replace function match_research_docs(query_embedding vector(1024), match_count int)
returns setof research_docs
language sql
as $$
  select * from research_docs
  order by embedding <-> query_embedding
  limit match_count;
$$;
