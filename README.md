# PRTECH Business OS

Multi-agent orchestration platform: a LangGraph supervisor routes requests to
specialized sub-agents (Lead-Gen, Outreach, Social Poster, Research,
Form-Fill, Monitor) that share a Supabase memory layer and a common
Playwright browser tool.

**Current status:** `router`, `Lead-Gen`, `Outreach`, `Research`, and
`Social Poster` are implemented end-to-end. Outreach and Social Poster both
default to **draft-only** (nothing sent/posted) unless explicitly opted
into auto-send/auto-post. Research paraphrases every source in its own
words (never quotes verbatim) and stores findings as embeddings in
`research_docs` for later semantic search — note: embedding storage is
currently failing (`stored_doc_ids` comes back empty) due to an NVIDIA NIM
embedding-model issue still being debugged; the report/sources themselves
work fine. `Form-Fill` and `Monitor` are stubbed (`NotImplementedError`) so
the graph shape is in place; build them incrementally per Step 3 of the
original build plan.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# fill in NVIDIA_API_KEY, GROQ_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
# NVIDIA_API_KEY: free key from https://build.nvidia.com
# GROQ_API_KEY: free key from https://console.groq.com
```

Then run `backend/schema.sql` in the Supabase SQL editor to create the
`leads`, `outreach_log`, `research_docs`, `monitor_snapshots`, and
`agent_audit_log` tables (and the pgvector extension + similarity RPC).

## Run

```bash
uvicorn main:app --reload --port 8000
```

Test the vertical slice:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "find dentists in Coimbatore"}'
```

## Important caveats before you rely on this

- **Scraping vs. ToS**: `agents/lead_gen.py`'s Google Maps scraper is a
  prototype with placeholder selectors that will break as the DOM changes,
  and browser-scraping Maps/JustDial may violate those sites' Terms of
  Service. For anything beyond local prototyping, swap `_search_google_maps`
  for the official Google Places API (see the docstring in that file).
- **Outreach = cold email at scale**: the spec calls for rate limiting
  (`MAX_OUTREACH_PER_HOUR`) and a human-approval draft-only default before
  auto-send. Don't flip auto-send on without also handling unsubscribe
  requests and complying with applicable anti-spam law (e.g. CAN-SPAM,
  India's IT Act / DPDP rules) for your jurisdiction and your recipients'.
- **Social automation**: posting/DM'ing via browser automation instead of
  official APIs is against Instagram's and LinkedIn's Terms of Service and
  can get the automating account banned. Prefer the official Graph API /
  LinkedIn API where available; treat browser-automation posting as a
  prototype fallback only.
- **Verification**: every browser action that changes external state should
  go through `BrowserTool.run_with_verification(...)`, which retries and
  can call an LLM self-check against a screenshot — wire your vision model
  of choice into the `self_check_fn` argument before relying on it.

## Project layout

```
backend/
├── main.py                  # FastAPI entrypoint (/chat, /leads, /outreach/log, /monitor/add, /research/{id})
├── orchestrator/
│   ├── supervisor.py        # LangGraph StateGraph wiring
│   └── router.py            # Groq/Llama intent classification
├── agents/
│   ├── lead_gen.py          # implemented
│   ├── outreach.py          # implemented (draft-only by default)
│   ├── social_poster.py     # implemented (draft-only by default)
│   ├── research.py          # implemented
│   ├── form_fill.py         # stub
│   └── monitor.py           # stub
├── tools/
│   ├── browser.py           # Playwright wrapper + retry/self-check
│   ├── llm.py                # NVIDIA NIM helper (free tier, planning/writing)
│   ├── email_sender.py       # SMTP sender used by Outreach
│   └── vector_store.py       # Supabase helpers (leads, outreach_log, research_docs, ...)
├── memory/
│   └── shared_state.py      # LangGraph state schema
├── schema.sql
├── requirements.txt
└── .env.example
```

## Next steps (Step 4-7 from the build plan)

1. Add a proper param-extraction step in front of `lead_gen`/`outreach`
   (currently a naive string split on `" in "` — swap for an LLM extraction
   call so multi-part requests like "email the dentists I found earlier"
   route correctly).
2. Build `form_fill.py` and `monitor.py`, each returning through the
   existing graph shape.
3. `research.py` now uses the Tavily Search API (free tier, 1,000
   queries/month, no card required) instead of scraping a search engine —
   set `TAVILY_API_KEY` in `.env`. Tavily returns extracted page content
   directly, so this agent doesn't use Playwright/the browser tool at all
   anymore.
4. `social_poster.py`'s auto-post path needs a real `SocialPoster`
   implementation before it can actually publish anything — wire in the
   Instagram Graph API and/or LinkedIn Marketing API (both free, but
   require going through each platform's app-review process). Do not
   implement auto-posting via browser automation; it violates both
   platforms' Terms of Service and risks the account being banned.
5. Add `agent_audit_log` writes to every agent node for debugging.
6. Build the Next.js chat UI against `/chat`.
7. Before flipping `auto_send=True` on Outreach anywhere: confirm you're
   complying with applicable anti-spam law for your jurisdiction and your
   recipients' (e.g. CAN-SPAM, India's IT Act / DPDP rules), and make sure
   `MAX_OUTREACH_PER_HOUR` is tuned to something your SMTP provider allows.
8. Debug why Research's embedding storage still fails (`stored_doc_ids`
   comes back empty) — check the uvicorn logs around the `nim_embed` call
   for the actual error; last known state was that `tools/llm.py` was
   updated to auto-discover a live embedding model but the fix wasn't
   confirmed working yet.
