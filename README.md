# PRTECH Business OS

Multi-agent orchestration platform: a LangGraph supervisor routes requests to
specialized sub-agents (Lead-Gen, Outreach, Social Poster, Research,
Form-Fill, Monitor) that share a Supabase memory layer and a common
Playwright browser tool.

For a full breakdown of how the system fits together — request lifecycle,
component responsibilities, data model, and the reasoning behind key design
decisions — see [ARCHITECTURE.md](./ARCHITECTURE.md).

**Current status:** all six agents (`Lead-Gen`, `Outreach`, `Research`,
`Social Poster`, `Form-Fill`, `Monitor`) plus `router`/`supervisor` are
implemented, tested live, and confirmed working end-to-end. Outreach and
Social Poster both default to **draft-only** (nothing sent/posted) unless
explicitly opted into auto-send/auto-post. Form-Fill defaults to **dry-run**
(fills but never submits). Research paraphrases every source in its own
words (never quotes verbatim), and its embeddings are stored in
`research_docs` for later semantic search — `tools/llm.py` auto-discovers a
live NIM model for both chat and embeddings, since NVIDIA has retired
several free-tier models mid-project, and any future embedding failure
surfaces directly in the API response via an `embedding_errors` field
instead of only in server logs. `lead_gen`/`outreach` use LLM-based param
extraction rather than naive string-splitting, so natural phrasing beyond
exact "X in Y" routes correctly. Lead-Gen sources data from OpenStreetMap
(free, no API key, no card) rather than scraping Google Maps, and can
optionally enrich leads missing phone/website via `POST /leads/enrich`.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# fill in NVIDIA_API_KEY, GROQ_API_KEY, TAVILY_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
# NVIDIA_API_KEY: free key from https://build.nvidia.com
# GROQ_API_KEY: free key from https://console.groq.com
# TAVILY_API_KEY: free key from https://app.tavily.com
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

Fill in missing phone/website on existing leads (OSM data often has name +
location but not contact details — see the caveat below):

```
curl -X POST http://localhost:8000/leads/enrich -H "Content-Type: application/json" -d "{\"niche\": \"dentists\", \"location\": \"Coimbatore\", \"max_leads\": 20}"
```

Every agent call writes a row to `agent_audit_log` — browse it (optionally filtered by agent):

```
curl http://localhost:8000/audit-log
curl "http://localhost:8000/audit-log?agent=lead_gen"
```

**If you're on Windows PowerShell** (not cmd.exe — check your prompt),
`curl` is aliased to `Invoke-WebRequest`/`Invoke-RestMethod` and needs
different syntax. The equivalent for any of the above:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method Post -ContentType "application/json" -Body '{"message": "find dentists in Coimbatore"}'
Invoke-RestMethod -Uri "http://localhost:8000/leads/enrich" -Method Post -ContentType "application/json" -Body '{"niche": "dentists", "location": "Coimbatore", "max_leads": 20}'
Invoke-RestMethod -Uri "http://localhost:8000/audit-log" -Method Get
```

## Smoke testing

Rather than manually re-running curl commands for every agent after a
dependency update or a provider retiring a model (which has happened
multiple times during this project's development — see ARCHITECTURE.md
§5.3), run the smoke test script against a live server:

```bash
# with uvicorn already running in another terminal
python scripts/smoke_test.py
```

Exercises all six agents plus `/audit-log`, checks response shapes (and,
critically, that Outreach/Social genuinely default to `draft_only` rather
than silently auto-sending), and prints a pass/fail summary. Exit code is 0
if everything passed, 1 otherwise — safe to use in a CI step, not just
interactively.

```bash
# skip the slower browser/network-heavy agents for a quick LLM-only check
python scripts/smoke_test.py --skip lead_gen,monitor,form_fill

# point at a different host/port
python scripts/smoke_test.py --base-url http://localhost:8080
```

## Important caveats before you rely on this

- **Lead data coverage**: `agents/lead_gen.py` uses OpenStreetMap
  (Nominatim for geocoding, Overpass API for business data) — genuinely
  free, no API key, no billing account required (unlike the official
  Google Places API, which now requires a billing-enabled account even for
  its free monthly allowance). Trade-off: OSM's business listings are
  volunteer-maintained and can be sparser than Google Maps in some
  regions — results depend on how well-mapped the target area is. Public
  Nominatim/Overpass instances are also rate-limited for heavy use; see
  the module docstring for self-hosting notes if you outgrow them.
  Contact fields (phone/website/email) are tagged far less consistently
  than name/location — use `POST /leads/enrich` to try to fill those gaps
  via web search, though it won't find everything either.
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
├── main.py                  # FastAPI entrypoint (/chat, /leads, /leads/enrich, /outreach/log, /monitor/add, /research/{id}, /audit-log)
├── orchestrator/
│   ├── supervisor.py        # LangGraph StateGraph wiring
│   ├── router.py            # Groq intent classification
│   ├── param_extraction.py  # NIM-based niche/location extraction (lead_gen, outreach)
│   └── audit_log.py         # writes every agent call to agent_audit_log
├── agents/
│   ├── lead_gen.py          # implemented (OpenStreetMap; includes enrich_leads)
│   ├── outreach.py          # implemented (draft-only by default)
│   ├── social_poster.py     # implemented (draft-only by default)
│   ├── research.py          # implemented
│   ├── form_fill.py         # implemented (dry-run by default)
│   └── monitor.py           # implemented (single on-demand check; no built-in scheduler)
├── tools/
│   ├── browser.py           # Playwright wrapper + retry/self-check
│   ├── browser_runner.py    # thread-isolated Playwright sessions (Windows compatibility)
│   ├── llm.py                # NVIDIA NIM helper (free tier, auto-discovers live models)
│   ├── email_sender.py       # SMTP sender used by Outreach and Monitor alerts
│   └── vector_store.py       # Supabase helpers (leads, outreach_log, research_docs, ...)
├── memory/
│   └── shared_state.py      # LangGraph state schema
├── scripts/
│   └── smoke_test.py        # end-to-end test against a live server (see Smoke testing above)
├── schema.sql
├── requirements.txt
└── .env.example
```

## Next steps (Step 4-7 from the build plan)

1. All six agents are implemented. `lead_gen`/`outreach` now use LLM-based
   param extraction (`orchestrator/param_extraction.py`, powered by NIM) to
   pull niche/location out of free-text messages — handles phrasing like
   "find me some dentists near Chennai" or "any plumbers around Bangalore?",
   not just the old exact "X in Y" pattern. Falls back to a naive string
   split if the NIM call ever fails, so a transient outage degrades
   gracefully instead of breaking the request. `research`/`social` pass the
   whole message through as-is (correct for those — no extraction needed).
   `form_fill`/`monitor` take structured `params` instead (see the Run
   section above).
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
4. `agent_audit_log` is now written by every agent — see
   `orchestrator/audit_log.py`'s `with_audit(...)` decorator, applied to
   each of the six agent nodes at graph-registration time (plus the direct
   `/monitor/add` and `/leads/enrich` endpoints, which bypass the graph).
   Browse it via `GET /audit-log` (optionally `?agent=lead_gen` etc).
5. Build the Next.js chat UI against `/chat`.
6. Before flipping `auto_send=True` on Outreach or `dry_run=False` on
   Form-Fill anywhere real: confirm you're complying with applicable
   anti-spam law for Outreach (e.g. CAN-SPAM, India's IT Act / DPDP rules),
   and that `MAX_OUTREACH_PER_HOUR` is tuned to something your SMTP
   provider allows.
7. If Research's `embedding_errors` field ever comes back non-empty, that
   tells you exactly which NIM embedding call failed and why — no more
   digging through server logs needed, the API response carries it now.
8. `POST /leads/enrich` uses Tavily search + NIM extraction and can be
   extended to run automatically after `lead_gen` finds new leads, rather
   than requiring a separate manual call.
