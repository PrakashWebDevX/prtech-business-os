"""
Shared Playwright wrapper used by every agent that needs to touch the web.

Design notes:
- One BrowserTool instance is meant to be reused across a single agent run
  (call `start()` once, `close()` when the run is done) rather than spinning
  up a fresh browser per action.
- `run_with_verification` wraps an action in retry logic + an LLM self-check
  against a screenshot, per the spec's Step 2/6 verification requirement.
- Kept dependency-light: only Playwright + an injected LLM callable for the
  self-check, so this file has no hard dependency on which LLM client you use.
"""

import asyncio
import base64
import logging
from typing import Awaitable, Callable, Optional

from playwright.async_api import Browser, Page, Playwright, async_playwright
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("prtech.browser")

# Signature for whatever LLM call you plug in to do the visual self-check.
# It receives (goal_description, screenshot_bytes) and must return True/False.
SelfCheckFn = Callable[[str, bytes], Awaitable[bool]]


class BrowserTool:
    def __init__(self, headless: bool = True, self_check_fn: Optional[SelfCheckFn] = None):
        self.headless = headless
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._self_check_fn = self_check_fn

    async def start(self) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        self.page = await self._browser.new_page()

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    # ---- primitive actions -------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def navigate(self, url: str) -> None:
        assert self.page is not None, "call start() first"
        await self.page.goto(url, wait_until="domcontentloaded", timeout=20_000)

    async def extract_text(self, selector: str) -> Optional[str]:
        assert self.page is not None
        el = await self.page.query_selector(selector)
        if not el:
            return None
        return (await el.text_content() or "").strip()

    async def extract_all(self, selector: str) -> list[str]:
        assert self.page is not None
        els = await self.page.query_selector_all(selector)
        out = []
        for el in els:
            text = (await el.text_content() or "").strip()
            if text:
                out.append(text)
        return out

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def click(self, selector: str) -> None:
        assert self.page is not None
        await self.page.click(selector, timeout=10_000)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def fill(self, selector: str, value: str) -> None:
        assert self.page is not None
        await self.page.fill(selector, value, timeout=10_000)

    async def screenshot(self) -> bytes:
        assert self.page is not None
        return await self.page.screenshot(type="png")

    # ---- verified action wrapper (Step 6: verification & safety) ----------

    async def run_with_verification(
        self,
        action: Callable[[], Awaitable[None]],
        goal_description: str,
        max_attempts: int = 3,
    ) -> bool:
        """
        Runs `action`, takes a screenshot, and (if a self-check fn was
        provided) asks an LLM whether the resulting page state matches
        `goal_description`. Retries up to `max_attempts` times.

        Returns True once verified, False if it never verifies.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                await action()
            except Exception as exc:  # noqa: BLE001 - log and retry
                last_error = exc
                logger.warning("Action attempt %s/%s failed: %s", attempt, max_attempts, exc)
                await asyncio.sleep(min(2 ** attempt, 8))
                continue

            shot = await self.screenshot()

            if self._self_check_fn is None:
                # No verifier configured: treat a non-throwing action as success.
                return True

            ok = await self._self_check_fn(goal_description, shot)
            if ok:
                return True
            logger.warning("Self-check failed on attempt %s/%s for goal: %s", attempt, max_attempts, goal_description)

        if last_error:
            logger.error("run_with_verification exhausted retries: %s", last_error)
        return False


def screenshot_to_data_url(png_bytes: bytes) -> str:
    """Helper for passing a screenshot to a vision-capable LLM call."""
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"
