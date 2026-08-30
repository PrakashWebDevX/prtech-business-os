"""
Outreach Agent

Input: reference to a lead batch (niche + location, or explicit lead_ids)
Behavior:
  1. Reads matching rows from the `leads` table.
  2. Drafts a personalized email per lead via NVIDIA NIM (tools/llm.py).
  3. In draft-only mode (the default): returns the drafts for human review,
     does NOT send anything, does NOT write to outreach_log.
  4. In auto-send mode (opt-in via auto_send=True): sends via SMTP, respects
     MAX_OUTREACH_PER_HOUR by pacing sends, and logs every attempt
     (sent or failed) to outreach_log.

Safety defaults, per the build plan's Step 6 (Verification & Safety):
  - auto_send defaults to False. The caller (supervisor graph / API) has to
    explicitly opt in.
  - Rate limiting is enforced in-process via a simple sleep-based pacer.
    For multi-worker deployments, replace this with a shared rate limiter
    (e.g. Redis token bucket) since an in-memory pacer only limits a single
    process.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

from tools.email_sender import send_email
from tools.llm import nim_complete
from tools.vector_store import insert_rows, select_rows

logger = logging.getLogger("prtech.agents.outreach")

_DRAFT_SYSTEM_PROMPT = """You are a sales outreach copywriter. Write a short, personalized,
non-spammy first-touch email to a local business, based on the business context given.

Rules:
- Keep it under 120 words.
- No hype, no exclamation points, no "Dear Sir/Madam".
- Reference the specific business by name and niche naturally.
- End with a low-pressure call to action (a question, not a demand).
- Output ONLY the email body text. No subject line, no markdown, no preamble.
"""


def _draft_email(lead: dict, offer_context: str) -> str:
    user_prompt = (
        f"Business name: {lead.get('business_name')}\n"
        f"Niche: {lead.get('niche')}\n"
        f"Location: {lead.get('location')}\n"
        f"Website: {lead.get('website') or 'unknown'}\n\n"
        f"What we're offering / why we're reaching out: {offer_context}\n\n"
        "Write the outreach email body now."
    )
    return nim_complete(_DRAFT_SYSTEM_PROMPT, user_prompt, temperature=0.6, max_tokens=300)


def _subject_line(lead: dict) -> str:
    return f"Quick question for {lead.get('business_name')}"


async def run_outreach(
    niche: str | None = None,
    location: str | None = None,
    lead_ids: list[str] | None = None,
    offer_context: str = "We help local businesses generate more leads online.",
    auto_send: bool = False,
    max_leads: int = 20,
) -> dict:
    if lead_ids:
        leads = []
        for lid in lead_ids:
            rows = select_rows("leads", filters={"id": lid}, limit=1)
            leads.extend(rows)
    else:
        filters = {}
        if niche:
            filters["niche"] = niche
        if location:
            filters["location"] = location
        leads = select_rows("leads", filters=filters, limit=max_leads)

    leads = [l for l in leads if l.get("email")]  # can only email leads with an email on file
    if not leads:
        return {"drafted": 0, "sent": 0, "failed": 0, "mode": "auto_send" if auto_send else "draft_only", "results": []}

    max_per_hour = int(os.environ.get("MAX_OUTREACH_PER_HOUR", "20"))
    delay_seconds = max(3600 / max_per_hour, 1) if auto_send else 0

    results = []
    log_rows = []
    sent_count = 0
    failed_count = 0

    for i, lead in enumerate(leads[:max_leads]):
        body = _draft_email(lead, offer_context)
        subject = _subject_line(lead)

        entry = {
            "lead_id": lead["id"],
            "business_name": lead.get("business_name"),
            "to": lead["email"],
            "subject": subject,
            "body": body,
            "status": "draft",
        }

        if auto_send:
            ok = send_email(lead["email"], subject, body)
            entry["status"] = "sent" if ok else "failed"
            sent_count += 1 if ok else 0
            failed_count += 0 if ok else 1

            log_rows.append(
                {
                    "lead_id": lead["id"],
                    "channel": "email",
                    "message": body,
                    "status": entry["status"],
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            if i < len(leads) - 1:
                await asyncio.sleep(delay_seconds)

        results.append(entry)

    if log_rows:
        insert_rows("outreach_log", log_rows)

    logger.info(
        "outreach: mode=%s drafted=%s sent=%s failed=%s",
        "auto_send" if auto_send else "draft_only",
        len(results),
        sent_count,
        failed_count,
    )

    return {
        "mode": "auto_send" if auto_send else "draft_only",
        "drafted": len(results),
        "sent": sent_count,
        "failed": failed_count,
        "results": results,
    }
