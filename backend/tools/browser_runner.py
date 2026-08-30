"""
Windows workaround for Playwright + uvicorn.

uvicorn forces the Selector event loop on Windows *before* it ever imports
your app code (it sets the loop policy, then starts the loop via
asyncio.run(), and only then imports main.py inside that already-running
loop). Selector event loops on Windows cannot spawn subprocesses
(asyncio.create_subprocess_exec raises NotImplementedError), but Playwright
launches its browser as a subprocess. Setting the event loop policy from
inside main.py is too late — the main loop already exists by then.

The fix: run any Playwright session in a separate, dedicated thread that
creates its own fresh event loop under the Proactor policy (which does
support subprocesses), and bridge the result back to the caller. This is a
no-op passthrough in the sense that it behaves identically on non-Windows
platforms, where the default loop already supports subprocess creation —
the thread hop is harmless there too, just unnecessary overhead.
"""

import asyncio
import sys
import threading
from typing import Any, Awaitable, Callable


def _thread_target(coro_fn: Callable[..., Awaitable[Any]], args: tuple, kwargs: dict, box: dict) -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        box["result"] = loop.run_until_complete(coro_fn(*args, **kwargs))
    except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's side, see below
        box["error"] = exc
    finally:
        loop.close()


async def run_playwright_task(coro_fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
    """
    Runs `coro_fn(*args, **kwargs)` — an async function that uses
    tools/browser.py — to completion in a dedicated thread with its own
    event loop. Use this to wrap any Playwright session that would
    otherwise run on an already-active asyncio loop (e.g. inside a
    FastAPI/uvicorn request handler).

    Example:
        async def _browse(niche, location):
            browser = BrowserTool()
            await browser.start()
            ...
            return result

        result = await run_playwright_task(_browse, niche, location)
    """
    box: dict = {}
    thread = threading.Thread(target=_thread_target, args=(coro_fn, args, kwargs, box), daemon=True)
    thread.start()
    # Wait for the thread without blocking this event loop.
    await asyncio.get_running_loop().run_in_executor(None, thread.join)
    if "error" in box:
        raise box["error"]
    return box.get("result")
