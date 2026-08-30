"""
Form-Fill Agent

Input: a target form URL + a list of records (rows), each a dict mapping
field name -> value, plus a mapping of field name -> CSS selector on the
target form.
Behavior:
  1. For each row, navigates to the form URL, fills each mapped field.
  2. In dry-run mode (the default): fills the form but does NOT click
     submit, and does NOT move to the next row automatically — lets you
     visually verify one filled form before trusting the mapping. Returns
     per-row status without submitting anything.
  3. In submit mode (opt-in via dry_run=False): fills and submits each row,
     logs success/failure per row.

Notes:
- The field->selector mapping is the caller's responsibility. Every target
  form has different field names/selectors, so there's no generic way to
  auto-discover them reliably — pass field_selectors explicitly, e.g.
  {"name": "#full-name", "email": "input[name=email]"}.
- Uses tools/browser_runner.py to isolate the Playwright session in a
  dedicated thread — same Windows-compatibility reason as lead_gen.py and
  research.py (see that module's docstring for why).
- No self-check/verification screenshot step is wired in by default; add
  one via BrowserTool.run_with_verification if you need stronger
  confidence that each submission actually succeeded (e.g. checking for a
  "thank you" page or success message) rather than just "no exception was
  thrown".
"""

import logging

from tools.browser import BrowserTool
from tools.browser_runner import run_playwright_task

logger = logging.getLogger("prtech.agents.form_fill")


async def _fill_one_row(
    browser: BrowserTool,
    form_url: str,
    row: dict,
    field_selectors: dict[str, str],
    submit_selector: str | None,
    dry_run: bool,
) -> dict:
    result = {"row": row, "status": "filled", "error": None}
    try:
        await browser.navigate(form_url)

        for field_name, value in row.items():
            selector = field_selectors.get(field_name)
            if not selector:
                logger.warning("form_fill: no selector mapped for field %r — skipping", field_name)
                continue
            await browser.fill(selector, str(value))

        if not dry_run:
            if not submit_selector:
                result["status"] = "failed"
                result["error"] = "submit_selector not provided but dry_run=False"
            else:
                await browser.click(submit_selector)
                result["status"] = "submitted"
    except Exception as exc:  # noqa: BLE001 - record per-row failure, keep going with other rows
        result["status"] = "failed"
        result["error"] = str(exc)
        logger.error("form_fill: row failed: %s | row=%r", exc, row)

    return result


async def _run_all_rows(
    form_url: str,
    rows: list[dict],
    field_selectors: dict[str, str],
    submit_selector: str | None,
    dry_run: bool,
    headless: bool,
) -> list[dict]:
    browser = BrowserTool(headless=headless)
    await browser.start()
    try:
        results = []
        for row in rows:
            r = await _fill_one_row(browser, form_url, row, field_selectors, submit_selector, dry_run)
            results.append(r)
        return results
    finally:
        await browser.close()


async def run_form_fill(
    form_url: str,
    rows: list[dict],
    field_selectors: dict[str, str],
    submit_selector: str | None = None,
    dry_run: bool = True,
    headless: bool = True,
) -> dict:
    """
    field_selectors: {"field_name_in_row": "css_selector_on_form"}
    submit_selector: CSS selector for the submit button. Required if
    dry_run=False.
    """
    if not dry_run and not submit_selector:
        return {
            "mode": "submit",
            "error": "submit_selector is required when dry_run=False",
            "results": [],
        }

    # Run the whole Playwright session in a dedicated thread+loop — see
    # tools/browser_runner.py for why this is necessary on Windows.
    results = await run_playwright_task(_run_all_rows, form_url, rows, field_selectors, submit_selector, dry_run, headless)

    succeeded = sum(1 for r in results if r["status"] in ("filled", "submitted"))
    failed = sum(1 for r in results if r["status"] == "failed")

    logger.info(
        "form_fill: form_url=%s mode=%s total=%s succeeded=%s failed=%s",
        form_url,
        "dry_run" if dry_run else "submit",
        len(results),
        succeeded,
        failed,
    )

    return {
        "mode": "dry_run" if dry_run else "submit",
        "form_url": form_url,
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }
