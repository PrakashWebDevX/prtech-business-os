"""
Monitor Agent

Input: a URL to watch + what to detect (currently: any text-content change
on the page — price/listing-specific diffing can be layered on top by
extracting a narrower selector before hashing).
Behavior:
  1. Fetches the current page text via the browser tool, hashes it.
  2. Compares against the most recent snapshot stored in
     `monitor_snapshots` for this URL.
  3. If the hash differs (or no prior snapshot exists), stores a new
     snapshot row and returns changed=True.
  4. If an alert email is configured, sends a notification on change.

Notes:
- This is a single on-demand check (`run_monitor_check`), not a scheduler.
  The build plan calls for running checks on a schedule (cron via FastAPI
  background task or an external scheduler) — wire that up separately,
  e.g. a periodic task that calls run_monitor_check for every URL
  registered via POST /monitor/add, since this module doesn't manage its
  own schedule.
- Hashing full page text means unrelated changes (ads, timestamps, view
  counters) will also trigger a "changed" result. For price-only or
  listing-only monitoring, extract a narrower CSS selector before hashing
  instead of the whole page body — pass `selector=` to scope it.
- Uses tools/browser_runner.py to isolate the Playwright session in a
  dedicated thread — same Windows-compatibility reason as lead_gen.py,
  research.py, and form_fill.py.
"""

import hashlib
import logging

from tools.browser import BrowserTool
from tools.browser_runner import run_playwright_task
from tools.email_sender import send_email
from tools.vector_store import insert_rows, select_rows

logger = logging.getLogger("prtech.agents.monitor")


async def _fetch_page_text(url: str, selector: str, headless: bool) -> str:
    browser = BrowserTool(headless=headless)
    await browser.start()
    try:
        await browser.navigate(url)
        text = await browser.extract_text(selector)
        return text or ""
    finally:
        await browser.close()


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def run_monitor_check(
    url: str,
    selector: str = "body",
    alert_email: str | None = None,
    headless: bool = True,
) -> dict:
    """
    Runs a single check for `url` right now. Call this repeatedly on a
    schedule (see module docstring) to actually monitor over time.
    """
    # Run the whole Playwright session in a dedicated thread+loop — see
    # tools/browser_runner.py for why this is necessary on Windows.
    current_text = await run_playwright_task(_fetch_page_text, url, selector, headless)
    current_hash = _hash_content(current_text)

    previous = select_rows("monitor_snapshots", filters={"url": url}, limit=1, order_by="captured_at", desc=True)
    previous_hash = previous[0]["content_hash"] if previous else None

    changed = previous_hash is not None and previous_hash != current_hash
    is_first_check = previous_hash is None

    insert_rows("monitor_snapshots", [{"url": url, "content_hash": current_hash}])

    if changed and alert_email:
        ok = send_email(
            alert_email,
            subject=f"Change detected: {url}",
            body_text=f"The page at {url} has changed since the last check.\n\nCheck it here: {url}",
        )
        if not ok:
            logger.error("monitor: failed to send alert email for %s", url)

    logger.info(
        "monitor: url=%s changed=%s first_check=%s",
        url,
        changed,
        is_first_check,
    )

    return {
        "url": url,
        "changed": changed,
        "first_check": is_first_check,
        "current_hash": current_hash,
        "previous_hash": previous_hash,
        "alerted": bool(changed and alert_email),
    }


async def run_monitor(*args, **kwargs) -> dict:
    """
    Entry point matching the other agents' signature for the supervisor
    graph. Delegates straight to run_monitor_check — kept as a thin alias
    so the graph node wiring stays consistent (agents.<name>.run_<name>)
    across all six agents.
    """
    return await run_monitor_check(*args, **kwargs)
