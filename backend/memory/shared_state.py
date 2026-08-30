"""
Cross-agent state schema shared by the LangGraph supervisor and every agent node.

This is intentionally a plain TypedDict (not a pydantic model) because LangGraph's
StateGraph reducers work most predictably against dict-like state objects.
"""

from typing import Any, List, Literal, Optional, TypedDict

Intent = Literal[
    "lead_gen",
    "outreach",
    "social",
    "research",
    "form_fill",
    "monitor",
    "clarify",
]


class AgentTurn(TypedDict, total=False):
    role: str  # "user" | "agent" | "system"
    content: str
    agent: Optional[str]


class SharedState(TypedDict, total=False):
    # Raw input for this turn
    user_input: str

    # Set by the router node
    intent: Optional[Intent]
    active_agent: Optional[str]

    # Set by whichever agent node runs
    agent_output: Optional[Any]

    # References into Supabase rows created/updated this turn, e.g.
    # {"leads": [<uuid>, ...], "research_docs": [<uuid>]}
    shared_memory_refs: dict

    # Rolling conversation history for multi-step requests
    history: List[AgentTurn]

    # Set to True by an agent node when it wants control routed back to
    # the router for a follow-up step (e.g. lead_gen -> outreach chaining)
    needs_followup: bool
