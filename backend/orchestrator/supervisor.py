"""
LangGraph supervisor graph.

router -> lead_gen, router -> outreach, router -> research, and
router -> social are fully implemented. The remaining two agents
(form_fill, monitor) are wired in as stub nodes so the graph shape is in
place; extend them as each is implemented in backend/agents/.

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
from orchestrator.router import router_node

logger = logging.getLogger("prtech.orchestrator")


async def _lead_gen_node(state: SharedState) -> SharedState:
    # Very small param extraction for the MVP — replace with a proper LLM
    # extraction step once the vertical slice is working end-to-end.
    user_input = state["user_input"]
    niche, _, location = user_input.partition(" in ")
    niche = niche.replace("find", "").replace("Find", "").strip() or user_input
    location = location.strip() or "unspecified"

    result = await lead_gen.run_lead_gen(niche, location)
    state["agent_output"] = result
    state.setdefault("shared_memory_refs", {})["leads"] = result.get("lead_ids", [])
    return state


async def _outreach_node(state: SharedState) -> SharedState:
    # MVP param extraction, same caveat as lead_gen: replace with a proper
    # LLM extraction step. Defaults to draft-only (auto_send=False) — the
    # supervisor never auto-sends on its own; that has to be requested
    # explicitly via the /chat payload or a follow-up confirmation step.
    user_input = state["user_input"]
    niche, _, location = user_input.partition(" in ")
    niche = niche.strip() or None
    location = location.strip() or None

    result = await outreach.run_outreach(niche=niche, location=location, auto_send=False)
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


def _not_implemented_node(agent_name: str, coro_fn):
    async def _node(state: SharedState) -> SharedState:
        try:
            await coro_fn()
        except NotImplementedError as exc:
            state["agent_output"] = {"error": str(exc), "agent": agent_name}
        return state

    return _node


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
    graph.add_node("form_fill", _not_implemented_node("form_fill", form_fill.run_form_fill))
    graph.add_node("monitor", _not_implemented_node("monitor", monitor.run_monitor))
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
