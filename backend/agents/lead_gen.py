"""
Lead-Gen Agent

Input: a niche + location string, e.g. "dentists in Coimbatore".
Output: a deduped list of leads written to the `leads` Supabase table.

Data source: OpenStreetMap's Nominatim (geocoding) + Overpass API (business
data) — both genuinely free, no API key, no signup, no billing account.

This replaced an earlier Google Maps browser-scraping prototype for two
reasons: (1) its CSS selectors were placeholders never verified against the
live DOM, and (2) more fundamentally, the free-tier replacement people
usually reach for — the official Google Places API — actually requires a
billing-enabled Google Cloud account (credit card on file) even to use its
free monthly allowance, which conflicts with this project's no-card
constraint. OpenStreetMap has no such requirement.

Trade-off to know about: OSM's business-listing coverage is volunteer-
maintained and can be sparser than Google Maps in some regions, especially
for small/informal businesses. Results will vary by how well-mapped the
target area is. In particular, contact fields (phone/website/email) are
tagged far less consistently than name/location — `enrich_leads` below
exists specifically to help fill those gaps.

Public Overpass/Nominatim instances are rate-limited (Nominatim: ~1
request/second; Overpass: a few hundred queries/day for moderate-sized
queries) — fine for interactive use, but a heavy production workload should
run against paid/self-hosted Overpass infrastructure instead of the public
endpoint used here.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from tools.llm import nim_complete
from tools.vector_store import insert_rows, select_rows, update_row

logger = logging.getLogger("prtech.agents.lead_gen")

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",  # fallback if the primary is down/rate-limiting
]

# Nominatim's usage policy requires a descriptive User-Agent identifying the
# application (not a generic library default) — this is a hard requirement,
# not a courtesy, and requests without one get blocked.
_USER_AGENT = "PRTECH-Business-OS/1.0 (lead-gen agent; contact: set-your-email-here)"

# Common niche phrasing -> OSM tag (key, value). Extend this as needed; OSM
# has hundreds of tag values, this covers the common lead-gen categories.
_NICHE_TAG_MAP: dict[str, tuple[str, str]] = {
    "dentist": ("amenity", "dentist"),
    "dentists": ("amenity", "dentist"),
    "doctor": ("amenity", "doctors"),
    "doctors": ("amenity", "doctors"),
    "clinic": ("amenity", "clinic"),
    "hospital": ("amenity", "hospital"),
    "pharmacy": ("amenity", "pharmacy"),
    "restaurant": ("amenity", "restaurant"),
    "restaurants": ("amenity", "restaurant"),
    "cafe": ("amenity", "cafe"),
    "cafes": ("amenity", "cafe"),
    "coffee shop": ("amenity", "cafe"),
    "bar": ("amenity", "bar"),
    "bakery": ("shop", "bakery"),
    "bakeries": ("shop", "bakery"),
    "salon": ("shop", "hairdresser"),
    "hair salon": ("shop", "hairdresser"),
    "barber": ("shop", "hairdresser"),
    "gym": ("leisure", "fitness_centre"),
    "fitness center": ("leisure", "fitness_centre"),
    "hotel": ("tourism", "hotel"),
    "hotels": ("tourism", "hotel"),
    "bank": ("amenity", "bank"),
    "lawyer": ("office", "lawyer"),
    "lawyers": ("office", "lawyer"),
    "law firm": ("office", "lawyer"),
    "accountant": ("office", "accountant"),
    "real estate": ("office", "estate_agent"),
    "real estate agent": ("office", "estate_agent"),
    "plumber": ("craft", "plumber"),
    "plumbers": ("craft", "plumber"),
    "electrician": ("craft", "electrician"),
    "electricians": ("craft", "electrician"),
    "supermarket": ("shop", "supermarket"),
    "grocery store": ("shop", "supermarket"),
    "school": ("amenity", "school"),
    "veterinarian": ("amenity", "veterinary"),
    "vet": ("amenity", "veterinary"),
}


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
    if lead.phone:
        return re.sub(r"\D", "", lead.phone)
    return f"{lead.business_name.strip().lower()}|{lead.location.strip().lower()}"


def _resolve_osm_tag(niche: str) -> tuple[str, str]:
    key = niche.strip().lower()
    if key in _NICHE_TAG_MAP:
        return _NICHE_TAG_MAP[key]
    # Loose fallback: strip trailing "s" for simple plurals not already listed.
    if key.endswith("s") and key[:-1] in _NICHE_TAG_MAP:
        return _NICHE_TAG_MAP[key[:-1]]
    # No mapping found — fall back to a generic "shop" tag with the niche as
    # the value, which OSM does use for many small-business categories
    # (shop=electronics, shop=furniture, etc.), though it won't match
    # every possible niche.
    logger.warning("lead_gen: no OSM tag mapping for niche %r — falling back to shop=%s", niche, key)
    return ("shop", key)


def _geocode_location(location: str) -> Optional[tuple[float, float]]:
    try:
        resp = httpx.get(
            _NOMINATIM_URL,
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": _USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.error("lead_gen: geocoding failed for %r: %s", location, exc)
        return None

    if not results:
        logger.warning("lead_gen: Nominatim found no match for location %r", location)
        return None

    return float(results[0]["lat"]), float(results[0]["lon"])


def _build_overpass_query(tag_key: str, tag_value: str, lat: float, lon: float, radius_m: int) -> str:
    around = f"(around:{radius_m},{lat},{lon})"
    return f"""
    [out:json][timeout:25];
    (
      node["{tag_key}"="{tag_value}"]{around};
      way["{tag_key}"="{tag_value}"]{around};
    );
    out center tags;
    """


def _run_overpass_query(query: str) -> list[dict]:
    last_error: Optional[Exception] = None
    for url in _OVERPASS_URLS:
        try:
            resp = httpx.post(url, data={"data": query}, headers={"User-Agent": _USER_AGENT}, timeout=30)
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            last_error = exc
            logger.warning("lead_gen: Overpass endpoint %s failed (%s), trying next", url, exc)
    logger.error("lead_gen: all Overpass endpoints failed: %s", last_error)
    return []


def _format_address(tags: dict) -> Optional[str]:
    parts = [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")]
    parts = [p for p in parts if p]
    return ", ".join(parts) if parts else None


def _elements_to_leads(elements: list[dict], niche: str, location: str, max_results: int) -> list[Lead]:
    leads = []
    for el in elements[:max_results]:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue  # unnamed OSM nodes aren't useful leads
        leads.append(
            Lead(
                niche=niche,
                location=location,
                business_name=name,
                phone=tags.get("phone") or tags.get("contact:phone"),
                email=tags.get("email") or tags.get("contact:email"),
                website=tags.get("website") or tags.get("contact:website"),
                socials={
                    k.split(":", 1)[1]: v
                    for k, v in tags.items()
                    if k.startswith("contact:") and k.split(":", 1)[1] in ("facebook", "instagram", "twitter")
                }
                or None,
            )
        )
    return leads


def _search_osm(niche: str, location: str, max_results: int, radius_m: int = 10_000) -> list[Lead]:
    coords = _geocode_location(location)
    if not coords:
        return []
    lat, lon = coords

    tag_key, tag_value = _resolve_osm_tag(niche)
    query = _build_overpass_query(tag_key, tag_value, lat, lon, radius_m)
    elements = _run_overpass_query(query)

    return _elements_to_leads(elements, niche, location, max_results)


async def run_lead_gen(niche: str, location: str, max_results: int = 20) -> dict:
    """
    Entry point called by the supervisor graph's lead_gen node.
    Returns a summary dict; also writes deduped rows to Supabase.

    Note: this is a sync HTTP call under the hood (httpx sync client), not
    Playwright — no thread-isolation workaround is needed here the way it
    is for lead_gen's old browser-based approach or for form_fill/monitor.
    """
    raw_leads = _search_osm(niche, location, max_results)

    existing = select_rows("leads", filters={"niche": niche, "location": location}, limit=1000)
    existing_keys = {
        _dedupe_key(Lead(niche=e["niche"], location=e["location"], business_name=e["business_name"], phone=e.get("phone")))
        for e in existing
    }

    new_leads = [l for l in raw_leads if _dedupe_key(l) not in existing_keys]

    inserted = insert_rows("leads", [asdict(l) for l in new_leads]) if new_leads else []

    logger.info(
        "lead_gen: niche=%r location=%r found=%s new=%s skipped_dupes=%s",
        niche,
        location,
        len(raw_leads),
        len(inserted),
        len(raw_leads) - len(new_leads),
    )

    return {
        "niche": niche,
        "location": location,
        "found": len(raw_leads),
        "inserted": len(inserted),
        "skipped_duplicates": len(raw_leads) - len(new_leads),
        "lead_ids": [row.get("id") for row in inserted],
    }


# --- Lead enrichment -------------------------------------------------------
#
# OSM listings frequently have a business's name and location but not its
# phone/website (see the module docstring's trade-off note). enrich_leads
# tries to fill those gaps by searching the web via Tavily (same free-tier
# API the Research agent uses) and asking NIM to extract a phone number and
# website URL from the results, then updating only the missing fields on
# the existing Supabase row — it never overwrites a field that already has
# a value.

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"

_ENRICH_SYSTEM_PROMPT = """You are extracting contact details for a business from web
search result snippets. You will be given the business name, its location, and several
raw text snippets that may or may not mention it.

Rules:
- Only extract a phone number or website if you are reasonably confident it belongs to
  THIS specific business at THIS location, not a different business with a similar name.
- phone: return in whatever format it appears, or null if not confidently found.
- website: return the business's own website URL, not a directory/listing page that
  merely mentions it (e.g. not a Yelp/JustDial/Facebook page unless that's genuinely
  the only web presence found) — null if not confidently found.
- If nothing reliable is found, return nulls rather than guessing.

Respond with ONLY a JSON object: {"phone": "<string or null>", "website": "<string or null>"}
No markdown, no code fences, no extra text.
"""


def _tavily_search_snippets(query: str, max_results: int = 3) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.error("lead_gen: TAVILY_API_KEY is not set — cannot enrich leads. Add it to .env.")
        return ""

    try:
        resp = httpx.post(
            _TAVILY_SEARCH_URL,
            json={"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.error("lead_gen: Tavily search failed during enrichment for %r: %s", query, exc)
        return ""

    results = data.get("results", [])
    return "\n\n".join(f"({r.get('url')}) {r.get('content', '')}" for r in results)


def _extract_contact_info(business_name: str, location: str, snippets: str) -> dict:
    if not snippets.strip():
        return {"phone": None, "website": None}

    user_prompt = f"Business: {business_name}\nLocation: {location}\n\nSearch snippets:\n{snippets}"
    try:
        # max_tokens is intentionally generous here (unlike the shorter
        # classification/extraction calls elsewhere in this codebase).
        # openai/gpt-oss-20b (NIM's current auto-selected chat model as of
        # this fix) is a reasoning model that spends tokens on internal
        # "thinking" before emitting its final answer — with this prompt's
        # longer input (multiple search snippets + a confidence judgment),
        # a 100-token budget was empirically observed to run out before any
        # visible output, returning an empty string 19/20 times in testing.
        raw = nim_complete(_ENRICH_SYSTEM_PROMPT, user_prompt, temperature=0, max_tokens=500)
        if not raw.strip():
            logger.warning(
                "lead_gen: contact extraction for %r got an empty response from the model "
                "(likely ran out of token budget on internal reasoning) — treating as not found",
                business_name,
            )
            return {"phone": None, "website": None}

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
        return {"phone": parsed.get("phone") or None, "website": parsed.get("website") or None}
    except Exception as exc:  # noqa: BLE001 - degrade to "found nothing" rather than fail the whole batch
        logger.warning("lead_gen: contact extraction failed for %r: %s", business_name, exc)
        return {"phone": None, "website": None}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
def _select_rows_with_retry(*args, **kwargs) -> list[dict]:
    return select_rows(*args, **kwargs)


async def enrich_leads(
    niche: Optional[str] = None,
    location: Optional[str] = None,
    lead_ids: Optional[list[str]] = None,
    max_leads: int = 20,
) -> dict:
    """
    Finds leads missing phone and/or website, tries to fill them in via
    Tavily search + NIM extraction, and updates only the missing fields —
    existing data is never overwritten. Returns a summary; does not raise
    on a per-lead failure, so one bad lookup doesn't stop the batch.

    The initial Supabase reads retry up to 3 times with backoff — a single
    transient DNS/network blip (which has happened repeatedly against
    Groq/Supabase during development on this network) shouldn't hard-fail
    the whole request.
    """
    if lead_ids:
        candidates = []
        for lid in lead_ids:
            rows = _select_rows_with_retry("leads", filters={"id": lid}, limit=1)
            candidates.extend(rows)
    else:
        filters = {}
        if niche:
            filters["niche"] = niche
        if location:
            filters["location"] = location
        candidates = _select_rows_with_retry("leads", filters=filters, limit=1000)

    # Only bother with leads actually missing something to fill in.
    candidates = [c for c in candidates if not c.get("phone") or not c.get("website")]
    candidates = candidates[:max_leads]

    if not candidates:
        return {"checked": 0, "enriched": 0, "still_missing": 0, "results": []}

    results = []
    enriched_count = 0

    for lead in candidates:
        query = f"{lead['business_name']} {lead['location']} phone number website contact"
        snippets = _tavily_search_snippets(query)
        found = _extract_contact_info(lead["business_name"], lead["location"], snippets)

        updates = {}
        if found["phone"] and not lead.get("phone"):
            updates["phone"] = found["phone"]
        if found["website"] and not lead.get("website"):
            updates["website"] = found["website"]

        entry = {"lead_id": lead["id"], "business_name": lead["business_name"], "updated_fields": list(updates.keys())}

        if updates:
            update_row("leads", lead["id"], updates)
            enriched_count += 1

        results.append(entry)

    logger.info(
        "lead_gen: enrich_leads checked=%s enriched=%s still_missing=%s",
        len(candidates),
        enriched_count,
        len(candidates) - enriched_count,
    )

    return {
        "checked": len(candidates),
        "enriched": enriched_count,
        "still_missing": len(candidates) - enriched_count,
        "results": results,
    }
