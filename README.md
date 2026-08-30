# PRTECH Business OS

Multi-agent orchestration platform: a LangGraph supervisor routes requests to
specialized sub-agents (Lead-Gen, Outreach, Social Poster, Research,
Form-Fill, Monitor) that share a Supabase memory layer and a common
Playwright browser tool.

**Current status:** all six agents (`Lead-Gen`, `Outreach`, `Research`,
`Social Poster`, `Form-Fill`, `Monitor`) plus `router`/`supervisor` are
implemented end-to-end. Outreach and Social Poster both default to
**draft-only** (nothing sent/posted) unless explicitly opted into
auto-send/auto-post. Form-Fill defaults to **dry-run** (fills but never
submits). Research paraphrases every source in its own words (never quotes
verbatim); embedding storage into `research_docs` was intermittently
failing due to NVIDIA NIM retiring embedding models — `tools/llm.py` now
auto-discovers a live one, and `run_research`'s response includes an
`embedding_errors` field so any future failure is visible directly in the
API response instead of only in server logs.

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

Test lead gen (Windows cmd.exe — escape quotes; use single-line curl, no `\` continuation):

```
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"find dentists in Coimbatore\"}"
```

`form_fill` and `monitor` need structured params, not just a free-text message — pass a `params` object:

```
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"fill the contact form\", \"params\": {\"form_url\": \"https://example.com/contact\", \"rows\": [{\"name\": \"Test User\", \"email\": \"test@example.com\"}], \"field_selectors\": {\"name\": \"#full-name\", \"email\": \"input[name=email]\"}, \"dry_run\": true}}"
```

```
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"watch this page\", \"params\": {\"url\": \"https://example.com/pricing\", \"alert_email\": \"you@example.com\"}}"
```

Or register a monitor directly (bypasses the router):

```
curl -X POST http://localhost:8000/monitor/add -H "Content-Type: application/json" -d "{\"url\": \"https://example.com/pricing\", \"alert_email\": \"you@example.com\"}"
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
│   ├── form_fill.py         # implemented (dry-run by default)
│   └── monitor.py           # implemented (single on-demand check; no built-in scheduler)
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

1. All six agents are implemented. `lead_gen`/`outreach`/`research`/`social`
   still parse params by crudely splitting the free-text `message` (e.g.
   `" in "` for niche/location) — swap for a proper LLM extraction step so
   multi-part requests like "email the dentists I found earlier" route
   correctly. `form_fill`/`monitor` already take structured `params`
   instead (see the Run section above), so they don't have this problem.
2. `monitor.py` only runs a single on-demand check — there's no built-in
   scheduler. Call `POST /monitor/add` (or route a `monitor` chat message)
   periodically via cron/an external scheduler to actually detect changes
   over time; each call after the first compares against the most recent
   stored snapshot.
3. `social_poster.py`'s auto-post path needs a real `SocialPoster`
   implementation before it can actually publish anything — wire in the
   Instagram Graph API and/or LinkedIn Marketing API (both free, but
   require going through each platform's app-review process). Do not
   implement auto-posting via browser automation; it violates both
   platforms' Terms of Service and risks the account being banned.
4. Add `agent_audit_log` writes to every agent node for debugging.
5. Build the Next.js chat UI against `/chat`.
6. Before flipping `auto_send=True` on Outreach or `dry_run=False` on
   Form-Fill anywhere real: confirm you're complying with applicable
   anti-spam law for Outreach (e.g. CAN-SPAM, India's IT Act / DPDP rules),
   and that `MAX_OUTREACH_PER_HOUR` is tuned to something your SMTP
   provider allows.
7. If Research's `embedding_errors` field ever comes back non-empty, that
   tells you exactly which NIM embedding call failed and why — no more
   digging through server logs needed, the API response carries it now.
