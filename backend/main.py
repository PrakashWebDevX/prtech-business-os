import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Windows only: harmless to set, but NOT sufficient on its own to fix
# Playwright + uvicorn on Windows — uvicorn forces its own event loop
# policy and creates the loop *before* this module is even imported, so
# this can't retroactively change the already-running main loop. The
# actual fix is tools/browser_runner.py, which runs any Playwright session
# in a separate thread with its own Proactor-policy loop. Left here mainly
# so any code path that creates a fresh loop later in this thread (rare)
# still gets the right default.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("prtech.main")

from orchestrator.supervisor import get_graph  # noqa: E402  (after load_dotenv)
from agents import monitor as monitor_agent  # noqa: E402
from tools.vector_store import select_rows  # noqa: E402

app = FastAPI(title="PRTECH Business OS")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None
    # Structured params for agents that can't reasonably be driven by free
    # text alone — form_fill needs {form_url, rows, field_selectors,
    # submit_selector?, dry_run?}; monitor needs {url, selector?,
    # alert_email?}. Omit for lead_gen/outreach/research/social, which
    # still parse from `message` (crudely — see supervisor.py's TODO on
    # replacing that with real LLM extraction).
    params: dict | None = None


class ChatResponse(BaseModel):
    intent: str | None
    agent_output: dict | None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    graph = get_graph()
    initial_state = {
        "user_input": req.message,
        "history": req.history or [],
        "shared_memory_refs": {},
        "params": req.params or {},
    }
    final_state = await graph.ainvoke(initial_state)
    return ChatResponse(
        intent=final_state.get("intent"),
        agent_output=final_state.get("agent_output"),
    )


@app.get("/leads")
async def get_leads(niche: str | None = None, location: str | None = None, limit: int = 100):
    filters = {}
    if niche:
        filters["niche"] = niche
    if location:
        filters["location"] = location
    return select_rows("leads", filters=filters, limit=limit)


@app.get("/outreach/log")
async def get_outreach_log(limit: int = 100):
    return select_rows("outreach_log", limit=limit)


class MonitorAddRequest(BaseModel):
    url: str
    selector: str = "body"
    alert_email: str | None = None


@app.post("/monitor/add")
async def add_monitor(req: MonitorAddRequest):
    """
    Registers a URL by running its first check immediately (which stores
    the baseline snapshot). There's no built-in scheduler yet — call this
    endpoint again periodically (e.g. from a cron job or external
    scheduler) to actually detect changes over time; each call after the
    first compares against the most recent stored snapshot.
    """
    return await monitor_agent.run_monitor_check(
        url=req.url,
        selector=req.selector,
        alert_email=req.alert_email,
    )


@app.get("/research/{doc_id}")
async def get_research_doc(doc_id: str):
    rows = select_rows("research_docs", filters={"id": doc_id}, limit=1)
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")
    return rows[0]


@app.get("/health")
async def health():
    return {"status": "ok"}
