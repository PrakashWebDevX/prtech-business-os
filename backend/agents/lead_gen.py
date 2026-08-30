"""
Lead-Gen Agent

Input: a niche + location string, e.g. "dentists in Coimbatore".
Output: a deduped list of leads written to the `leads` Supabase table.

IMPORTANT — read before wiring this up to a real target site:
Scraping Google Maps / JustDial via browser automation can violate those
sites' Terms of Service, and selectors below are placeholders that WILL
break as the DOM changes. For anything beyond a personal prototype, prefer
an official, ToS-compliant data source instead of scraping:
  - Google Places API (Text Search + Place Details) for Maps-equivalent data
  - JustDial does not offer a public leads API; scraping it is against their
    ToS, so treat that integration as prototype-only and swap it out.
This module is written so the scraping backend is a swappable function
(`_search_google_maps`) — replace its body with an official API call
without touching the rest of the agent.
"""

import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

from tools.browser import BrowserTool
from tools.browser_runner import run_playwright_task
from tools.vector_store import insert_rows, select_rows

logger = logging.getLogger("prtech.agents.lead_gen")


@dataclass
class Lead:
    niche: str
    location: str
    business_name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    socials: Optional[dict] = None
    score: Optional[float] = None


def _dedupe_key(lead: Lead) -> str:
    # Prefer phone as the dedup key (most stable), fall back to name+location.
    if lead.phone:
        return re.sub(r"\D", "", lead.phone)
    return f"{lead.business_name.strip().lower()}|{lead.location.strip().lower()}"


async def _search_google_maps(browser: BrowserTool, niche: str, location: str, max_results: int = 20) -> list[Lead]:
    """
    Placeholder scraping implementation. Selectors here are illustrative —
    Google Maps' DOM is not stable and changes without notice, so treat this
    as a starting point to adapt (or replace with the Places API call
    described in the module docstring) rather than production-ready code.
    """
    query = f"{niche} in {location}"
    search_url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
    await browser.navigate(search_url)

    # NOTE: replace these selectors with ones verified against the live DOM,
    # or better yet swap this whole function for a Places API call.
    result_cards = await browser.page.query_selector_all('div[role="feed"] > div')

    leads: list[Lead] = []
    for card in result_cards[:max_results]:
        name_el = await card.query_selector("div.fontHeadlineSmall")
        name = (await name_el.text_content()).strip() if name_el else None
        if not name:
            continue
        leads.append(Lead(niche=niche, location=location, business_name=name))

    return leads


async def _browse_leads(niche: str, location: str, max_results: int, headless: bool) -> list[Lead]:
    browser = BrowserTool(headless=headless)
    await browser.start()
    try:
        return await _search_google_maps(browser, niche, location, max_results=max_results)
    finally:
        await browser.close()


async def run_lead_gen(niche: str, location: str, max_results: int = 20, headless: bool = True) -> dict:
    """
    Entry point called by the supervisor graph's lead_gen node.
    Returns a summary dict; also writes deduped rows to Supabase.
    """
    # Run the whole Playwright session in a dedicated thread+loop — see
    # tools/browser_runner.py for why this is necessary on Windows.
    raw_leads = await run_playwright_task(_browse_leads, niche, location, max_results, headless)

    existing = select_rows("leads", filters={"niche": niche, "location": location}, limit=1000)
    existing_keys = {
        _dedupe_key(Lead(niche=e["niche"], location=e["location"], business_name=e["business_name"], phone=e.get("phone")))
        for e in existing
    }

    new_leads = [l for l in raw_leads if _dedupe_key(l) not in existing_keys]

    inserted = insert_rows("leads", [asdict(l) for l in new_leads]) if new_leads else []

    logger.info("lead_gen: found=%s new=%s skipped_dupes=%s", len(raw_leads), len(inserted), len(raw_leads) - len(new_leads))

    return {
        "niche": niche,
        "location": location,
        "found": len(raw_leads),
        "inserted": len(inserted),
        "skipped_duplicates": len(raw_leads) - len(new_leads),
        "lead_ids": [row.get("id") for row in inserted],
    }
