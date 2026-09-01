"""
LangGraph supervisor graph.

All six agents (lead_gen, outreach, research, social, form_fill, monitor)
are fully implemented and wired to real nodes.

IMPORTANT: every node here is `async def` and the graph is driven with
`ainvoke` (see main.py). Do NOT call `asyncio.run()` inside a node — FastAPI
endpoints are already running inside an event loop, and asyncio.run() raises
"cannot be called from a running event loop" if you try to start a second
one nested inside it. Async all the way through avoids that entirely.
"""

import logging

from langgraph.graph import END, StateGraph

from agents import form_fill, lead_gen, monitor, outreach, research, social_poster
from memory.shared_state import SharedState
from orchestrator.param_extraction import extract_niche_location
from orchestrator.router import router_node

logger = logging.getLogger("prtech.orchestrator")


async def _lead_gen_node(state: SharedState) -> SharedState:
    user_input = state["user_input"]
    extracted = extract_niche_location(user_input)
    niche = extracted["niche"] or user_input
    location = extracted["location"] or "unspecified"

    result = await lead_gen.run_lead_gen(niche, location)
    state["agent_output"] = result
    state.setdefault("shared_memory_refs", {})["leads"] = result.get("lead_ids", [])
    return state


async def _outreach_node(state: SharedState) -> SharedState:
    # Defaults to draft-only (auto_send=False) — the supervisor never
    # auto-sends on its own; that has to be requested explicitly via the
    # /chat payload or a follow-up confirmation step.
    user_input = state["user_input"]
    extracted = extract_niche_location(user_input)

    result = await outreach.run_outreach(niche=extracted["niche"], location=extracted["location"], auto_send=False)
    state["agent_output"] = result
    return state


async def _research_node(state: SharedState) -> SharedState:
    question = state["user_input"]
    result = await research.run_research(question)
    state["agent_output"] = result
    state.setdefault("shared_memory_refs", {})["research_docs"] = result.get("stored_doc_ids", [])
    return state


async def _social_node(state: SharedState) -> SharedState:
    # Treats the whole user_input as the content brief. auto_post is always
    # False here — the supervisor never auto-posts on its own, same
    # draft-only-by-default pattern as outreach.
    brief = state["user_input"]
    result = await social_poster.run_social_poster(brief=brief, auto_post=False)
    state["agent_output"] = result
    return state


_FORM_FILL_HELP = (
    "form_fill needs structured params — the /chat message alone isn't enough to fill a "
    "form safely. POST to /chat with a `params` field: "
    '{"form_url": "...", "rows": [{"name": "...", "email": "..."}], '
    '"field_selectors": {"name": "#full-name", "email": "input[name=email]"}, "dry_run": true}'
)


async def _form_fill_node(state: SharedState) -> SharedState:
    params = state.get("params") or {}
    if not params.get("form_url") or not params.get("rows") or not params.get("field_selectors"):
        state["agent_output"] = {"error": _FORM_FILL_HELP}
        return state

    result = await form_fill.run_form_fill(
        form_url=params["form_url"],
        rows=params["rows"],
        field_selectors=params["field_selectors"],
        submit_selector=params.get("submit_selector"),
        dry_run=params.get("dry_run", True),
    )
    state["agent_output"] = result
    return state


_MONITOR_HELP = (
    "monitor needs structured params — the /chat message alone isn't enough to identify "
    'a URL to watch. POST to /chat with a `params` field: {"url": "https://...", '
    '"selector": "body", "alert_email": "you@example.com"} (selector and alert_email are optional). '
    "For repeated checks over time, call POST /monitor/add on a schedule instead."
)


async def _monitor_node(state: SharedState) -> SharedState:
    params = state.get("params") or {}
    if not params.get("url"):
        state["agent_output"] = {"error": _MONITOR_HELP}
        return state

    result = await monitor.run_monitor_check(
        url=params["url"],
        selector=params.get("selector", "body"),
        alert_email=params.get("alert_email"),
    )
    state["agent_output"] = result
    return state


async def _clarify_node(state: SharedState) -> SharedState:
    state["agent_output"] = {
        "message": "I wasn't sure what you needed — could you clarify? "
        "For example: 'find dentists in Coimbatore' for leads, "
        "or 'research recent trends in X' for a report."
    }
    return state


def _route_from_router(state: SharedState) -> str:
    return state.get("intent") or "clarify"


def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(SharedState)

    graph.add_node("router", router_node)
    graph.add_node("lead_gen", _lead_gen_node)
    graph.add_node("outreach", _outreach_node)
    graph.add_node("social", _social_node)
    graph.add_node("research", _research_node)
    graph.add_node("form_fill", _form_fill_node)
    graph.add_node("monitor", _monitor_node)
    graph.add_node("clarify", _clarify_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "lead_gen": "lead_gen",
            "outreach": "outreach",
            "social": "social",
            "research": "research",
            "form_fill": "form_fill",
            "monitor": "monitor",
            "clarify": "clarify",
        },
    )

    for node_name in ["lead_gen", "outreach", "social", "research", "form_fill", "monitor", "clarify"]:
        graph.add_edge(node_name, END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_supervisor_graph()
    return _compiled_graph
