# PRTECH Business OS — Architecture

This document describes how the system fits together: the request lifecycle,
each component's responsibility, the data model, and the key design
decisions (and their trade-offs) made while building it. For setup and run
instructions, see [README.md](./README.md).

## 1. System overview

PRTECH Business OS is a **supervisor-and-agents** architecture built on
LangGraph. A single entry point (`POST /chat`) receives a free-text message
(plus optional structured `params`), a router classifies intent, and control
is handed to one of six specialized agents. Every agent shares the same
Supabase database for persistence; two of them (`form_fill`, `monitor`)
also share a Playwright-based browser tool.

```
                         ┌─────────────────────────────────────────┐
                         │              FastAPI (main.py)           │
                         │  POST /chat        POST /monitor/add      │
                         │  POST /leads/enrich GET  /leads            │
                         │  GET  /outreach/log GET  /research/{id}    │
                         │  GET  /audit-log                           │
                         └───────────────────┬───────────────────────┘
                                              │
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │        LangGraph Supervisor Graph         │
                         │           (orchestrator/supervisor.py)    │
                         └───────────────────┬───────────────────────┘
                                              │
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │              router_node                  │
                         │        (orchestrator/router.py)            │
                         │   Groq (openai/gpt-oss-20b) classifies     │
                         │   intent → lead_gen | outreach | research  │
                         │            | social | form_fill | monitor  │
                         │            | clarify                       │
                         └───────────────────┬───────────────────────┘
                                              │  conditional edge
              ┌───────────┬───────────┬──────┴──────┬───────────┬───────────┐
              ▼           ▼           ▼              ▼           ▼           ▼
         ┌─────────┐ ┌─────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐ ┌─────────┐
         │lead_gen │ │outreach │ │ research │  │  social  │ │form_fill │ │ monitor │
         └────┬────┘ └────┬────┘ └────┬─────┘  └────┬─────┘ └────┬─────┘ └────┬────┘
              │           │           │             │            │            │
              │  (each node wrapped in orchestrator/audit_log.py's with_audit)│
              └───────────┴─────┬─────┴─────────────┴────────────┴────────────┘
                                 ▼
                    ┌─────────────────────────┐        ┌──────────────────────┐
                    │   Supabase (Postgres      │◄──────┤  tools/vector_store.py │
                    │   + pgvector)              │       └──────────────────────┘
                    └─────────────────────────┘
                                 ▲
                    ┌────────────┴────────────┐
                    │  tools/browser.py         │  (form_fill, monitor only)
                    │  tools/browser_runner.py   │  thread-isolated Playwright
                    │  tools/llm.py               │  NVIDIA NIM (chat + embeddings)
                    │  tools/email_sender.py      │  SMTP (outreach, monitor alerts)
                    └────────────────────────────┘
```

Note: `lead_gen` (OpenStreetMap) and `research` (Tavily) do **not** use
Playwright — both moved off browser automation onto real APIs during
development (see §5.5 and §5.10). Only `form_fill` and `monitor` still need
a real browser, since target forms and monitored pages are often
JS-rendered.

## 2. Request lifecycle

Every `POST /chat` call follows the same path:

1. **`main.py`** receives `{message, history?, params?}`, builds the
   LangGraph initial state, and calls `graph.ainvoke(...)`.
2. **`router_node`** (`orchestrator/router.py`) sends the message to Groq
   (`openai/gpt-oss-20b`) with a system prompt listing the seven possible
   intents, and parses the JSON response. On any parse failure it logs the
   raw model output and falls back to `clarify` rather than guessing.
3. **Conditional edge** routes to the matching agent node based on the
   classified intent.
4. **The agent node** (in `orchestrator/supervisor.py`) does one of two
   things depending on the agent:
   - **`lead_gen` / `outreach`**: calls `orchestrator/param_extraction.py`
     to pull structured params (niche, location) out of the free-text
     message via an NVIDIA NIM call, with a naive-string-split fallback if
     that call fails.
   - **`research` / `social`**: passes the message through as-is (a
     research question or content brief doesn't need structured
     extraction).
   - **`form_fill` / `monitor`**: reads directly from the `params` field on
     the request — these need real structured input (a form URL + field
     selectors, or a URL to watch) that free text can't reliably carry, so
     the node returns a help message telling the caller what shape of
     `params` it expects if none was provided.
5. **`with_audit(agent_name)`** (`orchestrator/audit_log.py`) wraps every
   agent node at registration time, logging the call's input/output/success
   to `agent_audit_log` regardless of which agent ran.
6. **The agent itself** (in `agents/*.py`) does the actual work — searching,
   calling an LLM, reading/writing Supabase, browsing where needed — and
   returns a plain dict.
7. **The graph terminates** at `END`, and `main.py` returns
   `{intent, agent_output}` as the HTTP response.

Every node in the graph is `async def`, and the graph is driven with
`ainvoke` rather than `invoke` — see §5.1 for why this matters.

Two capabilities bypass the router entirely via dedicated endpoints, since
they operate on existing data rather than being naturally triggered by a
free-text message: `POST /monitor/add` (register/re-check a URL) and
`POST /leads/enrich` (fill missing contact fields on an existing lead
batch). Both still write to `agent_audit_log` via a direct call to
`log_agent_action(...)`, since there's no graph node to wrap for them.

## 3. Component responsibilities

### 3.1 `main.py` — API layer
FastAPI app exposing:
- `POST /chat` — the main router entry point
- `POST /monitor/add` — runs a Monitor check directly (bypasses intent
  classification; also how you'd trigger repeated checks from a scheduler)
- `POST /leads/enrich` — runs Lead-Gen's contact-info enrichment directly
  on an existing lead batch (see §3.3 and §5.11)
- `GET /leads`, `GET /outreach/log`, `GET /research/{id}`, `GET /audit-log`
  — read endpoints for agent-produced data
- `GET /health` — liveness check

### 3.2 `orchestrator/` — routing and coordination
- **`supervisor.py`** — builds the `StateGraph`: one node per agent plus
  `router` and `clarify`, wired with conditional edges from `router`'s
  classified intent.
- **`router.py`** — intent classification via Groq. Chosen for speed/cost
  over the planning model, since this is a cheap categorical decision, not
  a generation task.
- **`param_extraction.py`** — turns free text into structured
  `{niche, location}` for `lead_gen`/`outreach` via NVIDIA NIM. Exists
  because a naive `" in "` string split only handled one exact phrasing
  ("X in Y") and broke on anything else ("find me some dentists near X",
  "any plumbers around Y?").
- **`audit_log.py`** — `with_audit(agent_name)` decorator, applied to each
  of the six agent nodes at graph-registration time, writing every call's
  input/output/success to `agent_audit_log` (see §5.9).

### 3.3 `agents/` — the six domains
Each agent is a single `agents/<name>.py` module exposing an
`async def run_<name>(...)` entry point with a domain-specific signature
(not a shared interface — a lead-gen call and a monitor check take
genuinely different arguments). All six share Supabase and, where relevant,
the browser tool and the NIM client. `lead_gen.py` additionally exposes
`enrich_leads(...)`, a second entry point for filling missing contact
fields on leads already in the database (see §5.11).

| Agent | Default mode | External dependencies |
|---|---|---|
| `lead_gen` | — | OpenStreetMap (Nominatim geocoding + Overpass API); enrichment uses Tavily + NIM |
| `outreach` | **draft-only** | NIM (drafting), SMTP (send) |
| `research` | — | Tavily (search), NIM (paraphrase + embed) |
| `social_poster` | **draft-only** | NIM (drafting); no publish backend wired in |
| `form_fill` | **dry-run** | Playwright |
| `monitor` | — (single check, no scheduler) | Playwright, SMTP (alerts) |

### 3.4 `tools/` — shared infrastructure
- **`browser.py`** — Playwright wrapper: `navigate`, `extract_text`,
  `extract_all`, `click`, `fill`, `screenshot`, plus
  `run_with_verification(...)` for retry + optional LLM self-check against
  a screenshot before marking a browser action complete. Used only by
  `form_fill` and `monitor` — see §3's note in the overview diagram.
- **`browser_runner.py`** — runs a Playwright session in a dedicated
  thread with its own event loop. Exists specifically for Windows (see
  §5.2) but is a no-op-equivalent passthrough elsewhere.
- **`llm.py`** — NVIDIA NIM client wrapper. Auto-discovers a live chat
  model and a live embedding model via NIM's `/v1/models` endpoint rather
  than hardcoding model strings (see §5.3).
- **`email_sender.py`** — plain SMTP sender used by `outreach` (sending
  drafted emails) and `monitor` (change alerts).
- **`vector_store.py`** — Supabase helpers: generic `insert_rows` /
  `select_rows` / `upsert_rows` / `update_row`, plus `insert_research_doc`
  and `match_research_docs` (pgvector similarity search) for the Research
  agent's memory. `update_row` exists specifically for lead enrichment
  (updating an existing row's phone/website without touching other fields).

### 3.5 `memory/shared_state.py` — the graph's state schema
A `TypedDict` (`SharedState`) carrying `user_input`, `intent`,
`active_agent`, `agent_output`, `shared_memory_refs` (IDs of Supabase rows
written this turn), `history`, and `params` through the graph. LangGraph
threads this dict through every node; each node reads what it needs and
writes `agent_output` before the graph terminates.

## 4. Data model (Supabase / Postgres + pgvector)

```sql
leads              -- lead_gen writes; enrich_leads updates; outreach reads
outreach_log       -- outreach writes (only when auto_send=True)
research_docs      -- research writes (embedding vector(2048), see §5.4)
monitor_snapshots  -- monitor reads/writes every check
agent_audit_log     -- written by orchestrator/audit_log.py's with_audit(...) decorator, applied to every agent node
```

Full definitions are in `backend/schema.sql`, which must be run manually in
the Supabase SQL editor — there's no migration tooling, since the project
targets a single Supabase project per deployment rather than multi-environment
migrations.

`research_docs.embedding` and `match_research_docs`'s parameter are
`vector(2048)`, matching NVIDIA NIM's `nemotron-3-embed-1b` model's actual
output dimension — not a documented value, confirmed empirically (see
§5.4).

## 5. Key design decisions

### 5.1 Async all the way through
Every LangGraph node is `async def`, and `main.py` calls `graph.ainvoke(...)`
rather than the sync `graph.invoke(...)`. Early versions had nodes calling
`asyncio.run(...)` internally to bridge into async agent code — this breaks
inside a FastAPI request handler, which is already running inside an event
loop (`asyncio.run()` cannot start a second loop nested inside an existing
one). Making every node natively async removes the need for any such bridge.

### 5.2 Thread-isolated Playwright sessions (Windows compatibility)
On Windows, uvicorn's default event loop (Selector) cannot spawn
subprocesses, but Playwright launches its browser as a subprocess. Critically,
uvicorn sets its loop policy and creates the loop *before* importing the
FastAPI app — so setting `asyncio.set_event_loop_policy(...)` inside
`main.py` is too late to change the already-running main loop.

The fix (`tools/browser_runner.py`): every Playwright session runs inside a
dedicated background thread that creates its *own* fresh event loop under
the Proactor policy (which does support subprocess creation), and the result
is bridged back to the caller via `run_in_executor`. This is transparent to
callers — `agents/form_fill.py` and `agents/monitor.py` just wrap their
browsing logic in an inner `async def` and call
`await run_playwright_task(inner_fn, ...)` instead of running it directly.

`agents/lead_gen.py` and `agents/research.py` sidestep this issue entirely
by not using Playwright at all — see §5.5 and §5.10.

### 5.3 Model auto-discovery instead of hardcoded model strings
Both NVIDIA NIM and Groq have retired specific free-tier models multiple
times during this project's development (`meta/llama-3.1-70b-instruct`,
`meta/llama-3.3-70b-instruct`, and `nvidia/nv-embedqa-e5-v5` all went
`HTTP 410 Gone` within the same development window). Hardcoding a model
string means the whole app breaks every time the provider retires whatever
was picked.

`tools/llm.py` instead calls NIM's `/v1/models` endpoint once per process
(cached after first call) and picks the first live match from a preference
list, for both the chat model and the embedding model. An explicit
`NVIDIA_MODEL` / `NVIDIA_EMBED_MODEL` env var is tried first if set, but
only used if it's confirmed still live — otherwise auto-discovery takes
over rather than hard-failing.

### 5.4 Embedding dimension mismatches are caught explicitly
Different embedding models output different vector dimensions, and
Supabase's `vector(N)` column type is fixed at table-creation time. Rather
than let a dimension mismatch surface as an opaque Postgres insert error,
`nim_embed()` checks the actual returned vector length against the expected
dimension and raises an error containing the *exact* SQL (`alter table ...
alter column ... type vector(N)` plus the matching `match_research_docs`
function) needed to fix it. This is also why `research.py`'s response
includes an `embedding_errors` field — so a failure is visible in the API
response itself, not just in server logs.

This is exactly how the project discovered that `nemotron-3-embed-1b`
actually outputs 2048-dim vectors (not documented anywhere) rather than
the 1024-dim the schema originally assumed based on the retired
`nv-embedqa-e5-v5` model.

### 5.5 Research uses a real search API, not scraped search results
Earlier versions of `agents/research.py` scraped DuckDuckGo's HTML results
page with Playwright. This reliably hit DuckDuckGo's bot-detection CAPTCHA
("select all squares with a duck") — headless browsers are specifically
what that challenge is designed to block, so no amount of selector-fixing
would have made it reliable.

The fix: Tavily's Search API (free tier, 1,000 queries/month, no card
required), which also returns extracted page content directly in its
response — meaning `research.py` doesn't need Playwright *at all* anymore,
removing an entire category of fragility (and the Windows threading
workaround) from that one agent.

### 5.6 Draft-only / dry-run defaults on anything that acts externally
`outreach` (`auto_send`), `social_poster` (`auto_post`), and `form_fill`
(`dry_run`) all default to the safe, non-destructive mode. The supervisor
graph never flips these to the "live" mode on its own — doing so requires
an explicit `params`/argument override from the caller. This mirrors the
original build plan's Step 6 safety requirement and means routing a chat
message through the wrong intent by accident can't itself cause an email to
send, a post to publish, or a form to submit.

### 5.7 Social posting has no auto-publish backend by design
`agents/social_poster.py` defines a `SocialPoster` abstract interface for
the actual publish step, but ships no implementation — `_NotConfiguredPoster`
raises a clear `NotImplementedError` if `auto_post=True` is ever set without
one configured. This is deliberate: posting or DM'ing via browser automation
instead of the official Instagram Graph API / LinkedIn API violates both
platforms' Terms of Service and risks the automating account being banned.
Wiring up real auto-posting means implementing `SocialPoster` against
whichever official API you're using — see the module's docstring.

### 5.8 Copyright-safe research summarization
`agents/research.py`'s paraphrase prompt explicitly forbids verbatim
quoting or close paraphrasing of source text — findings are summarized
strictly in the researcher's own words, never extracted. This isn't a
nice-to-have; reproducing substantial chunks of someone else's article text
is a copyright problem regardless of downstream use, so it's enforced at
the prompt level rather than left to chance.

### 5.9 Audit logging as a decorator, not per-agent boilerplate
`orchestrator/audit_log.py`'s `with_audit(agent_name)` wraps a node
function at graph-registration time in `supervisor.py`, rather than each of
the six agent nodes calling a logging function internally. This guarantees
identical audit coverage across all six agents (input snapshot, resulting
output, a derived success/failure flag, and — for unhandled exceptions —
the exception message) without six near-duplicate logging blocks, and means
a future seventh agent gets audit logging for free just by using the same
wrapper. The two call sites that bypass the graph entirely
(`POST /monitor/add`, `POST /leads/enrich`) log directly instead, since
there's no node to wrap. A failure to *write* an audit row never breaks the
actual request — it's diagnostic infrastructure, not a critical path.

### 5.10 Lead-Gen uses OpenStreetMap, not Google Maps or the Places API
Earlier versions of `agents/lead_gen.py` scraped Google Maps with
Playwright, using placeholder CSS selectors never verified against the
live DOM. The obvious "proper" fix — the official Google Places API —
turned out not to fit this project's constraints: Google now requires a
billing-enabled Cloud account (a credit card on file) even to use its free
monthly per-SKU allowance, having removed the old card-free free tier.

The fix: OpenStreetMap's Nominatim (geocoding) and Overpass (business/POI
data) APIs — both genuinely free, no API key, no signup, no billing
account of any kind. This also means `lead_gen.py` no longer needs
Playwright at all (same simplification as Research's move to Tavily),
removing another category of fragility and the Windows-threading
workaround from this agent entirely. The trade-off: OSM's listings are
volunteer-maintained and can be sparser than Google's in less-mapped
regions, and niche→OSM-tag mapping is necessarily a curated lookup table
(`_NICHE_TAG_MAP`) rather than free-text search, so uncommon business
categories may need a mapping added.

### 5.11 Lead enrichment as a separate, explicit step
OSM listings frequently carry a business's name and location reliably but
not its phone or website — contact fields are tagged far less consistently
by whoever mapped the entry. Rather than trying to guess contact info at
scrape time, `enrich_leads()` (exposed via `POST /leads/enrich`, not routed
through `/chat`) is a deliberately separate, explicit step: find leads
missing phone/website, search the web via Tavily, ask NIM to extract a
confident phone/website match, and update *only* the missing fields —
never overwriting data that's already there. It's a direct endpoint rather
than a `/chat` intent because it operates on an existing lead batch (a
maintenance/backfill action) rather than being something a natural-language
message should trigger on its own.

## 6. Known limitations / prototype-grade pieces

- **`lead_gen.py`'s OSM data coverage varies by region** — OpenStreetMap's
  business listings are volunteer-maintained, so results can be sparser
  than Google Maps in less-mapped areas. The niche→OSM-tag mapping
  (`_NICHE_TAG_MAP`) also only covers common categories; an unmapped niche
  falls back to a generic `shop=<niche>` tag guess, which won't match
  every possible business type. `enrich_leads()` helps with contact-field
  gaps but is itself best-effort — Tavily searches don't always surface a
  confident phone/website match.
- **`monitor.py` has no built-in scheduler** — `run_monitor_check` performs
  one check per call. Real monitoring-over-time requires calling
  `POST /monitor/add` repeatedly via cron or an external scheduler.
- **No frontend** — `frontend/` is an empty placeholder; the original plan
  called for a Next.js chat UI against `/chat`, not yet built.
- **Single Supabase project, no migrations** — `schema.sql` is applied by
  hand; there's no versioned migration history.

## 7. Why this stack

- **LangGraph over a hand-rolled state machine**: conditional routing based
  on classified intent, with each agent as an isolated node, maps directly
  onto LangGraph's `StateGraph` primitives without extra scaffolding.
- **Groq for routing, NVIDIA NIM for generation**: intent classification is
  a fast, cheap, low-stakes categorical decision — Groq's inference speed
  fits that. Drafting emails, posts, and paraphrasing research findings are
  generation tasks that benefit from a stronger model — NIM's free-tier
  catalog includes larger models suited to that.
- **Playwright over `requests`/`BeautifulSoup`** for the two remaining
  browsing agents (`form_fill`, `monitor`): most real-world target forms
  and many monitored pages are JS-rendered, so a real browser is required,
  not just an HTTP client. `lead_gen` and `research` both moved off
  Playwright entirely once real APIs (OpenStreetMap, Tavily) covered their
  needs — see §5.5 and §5.10.
- **Supabase over a self-hosted Postgres**: free tier, built-in pgvector
  support (no separate vector DB needed for Research's semantic memory),
  and a REST API via `postgrest` that avoids needing a persistent DB
  connection pool from a request-scoped FastAPI process.
