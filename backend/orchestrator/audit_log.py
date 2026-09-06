"""
Audit logging for agent actions.

Writes to the `agent_audit_log` table (defined in schema.sql, per the
original build plan's Step 6: "Log every agent action to a shared audit
table for debugging"). The table existed from the start but nothing wrote
to it — this closes that gap.

Design: `with_audit(agent_name)` wraps a LangGraph node function rather than
requiring each of the six agent nodes to call a logging function manually.
This guarantees every agent gets identical, consistent audit coverage
without six near-duplicate blocks of logging code scattered through
supervisor.py, and means a seventh agent added later gets audit logging for
free just by using the same wrapper at registration time.

A failure to WRITE an audit row never breaks the actual request — audit
logging is diagnostic infrastructure, not a critical path. If Supabase is
briefly unreachable, the agent's real result still returns to the caller;
only the audit trail entry for that one call is missing.
"""

import logging
from functools import wraps
from typing import Awaitable, Callable

from memory.shared_state import SharedState
from tools.vector_store import insert_rows

logger = logging.getLogger("prtech.orchestrator.audit_log")

NodeFn = Callable[[SharedState], Awaitable[SharedState]]


def _looks_like_failure(output: object) -> bool:
    """
    Heuristic: every agent that hits a handled failure path in this
    codebase returns a dict with an "error" key (form_fill/monitor's
    missing-params messages, social_poster's NotImplementedError capture,
    etc.) rather than raising. An unhandled exception is caught separately
    below and always logged as a failure.
    """
    return isinstance(output, dict) and "error" in output


def log_agent_action(agent: str, action: str, input_data: dict, output_data: object, success: bool) -> None:
    try:
        insert_rows(
            "agent_audit_log",
            [
                {
                    "agent": agent,
                    "action": action,
                    "input": input_data,
                    "output": output_data if isinstance(output_data, (dict, list)) else {"result": str(output_data)},
                    "success": success,
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - audit logging must never break the actual request
        logger.error("audit_log: failed to write audit row for agent=%s action=%s: %s", agent, action, exc)


def with_audit(agent_name: str):
    """
    Decorator for a LangGraph node function. Logs the node's user_input,
    params, resulting agent_output, and a derived success/failure flag to
    agent_audit_log, then returns the node's result unchanged. Re-raises
    any exception from the wrapped node after logging it as a failure —
    this decorator observes, it never swallows errors.
    """

    def decorator(node_fn: NodeFn) -> NodeFn:
        @wraps(node_fn)
        async def wrapper(state: SharedState) -> SharedState:
            input_snapshot = {
                "user_input": state.get("user_input"),
                "params": state.get("params"),
            }
            try:
                result_state = await node_fn(state)
                output = result_state.get("agent_output")
                log_agent_action(
                    agent=agent_name,
                    action="run",
                    input_data=input_snapshot,
                    output_data=output,
                    success=not _looks_like_failure(output),
                )
                return result_state
            except Exception as exc:  # noqa: BLE001 - log the failure, then re-raise unchanged
                log_agent_action(
                    agent=agent_name,
                    action="run",
                    input_data=input_snapshot,
                    output_data={"exception": str(exc)},
                    success=False,
                )
                raise

        return wrapper

    return decorator
