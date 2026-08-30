"""
Research Agent

Input: a research question, e.g. "recent trends in local SEO for dentists".
Behavior:
  1. Calls the Tavily Search API (free tier: 1,000 queries/month, no card
     required — https://app.tavily.com) to get result URLs *and* extracted
     page content in one call. Tavily is purpose-built for this (AI/agent
     research use cases), so unlike a raw search API it saves us a separate
     page-fetch step entirely — no Playwright needed for this agent.
     (Earlier prototypes tried scraping DuckDuckGo directly, which got
     blocked by their bot-detection CAPTCHA, and then tried the Brave
     Search API, which is free but can prompt for card verification in
     some regions. Tavily's free tier needs neither a browser nor a card.)
  2. For each result, asks NVIDIA NIM to paraphrase the key findings in its
     own words — never verbatim extraction. This isn't optional: reproducing
     substantial chunks of someone else's article text is a copyright
     problem regardless of how the output is used downstream.
  3. Embeds each paraphrased finding via NIM and stores it in the
     `research_docs` pgvector table for later semantic search.
  4. Synthesizes a short structured report (own words) with source links.

Notes:
- This agent deliberately never asks the LLM to "quote" or "extract exact
  text" — the prompt only asks for paraphrased findings, by design.
- Tavily's `content` field is already a cleaned extract, not full raw HTML,
  so paraphrasing quality depends on how much of the source page Tavily's
  extractor captured — fine for a summary-level research agent like this.
"""

import logging
import os
from dataclasses import dataclass

import httpx

from tools.llm import nim_complete, nim_embed
from tools.vector_store import insert_research_doc

logger = logging.getLogger("prtech.agents.research")

_PARAPHRASE_SYSTEM_PROMPT = """You are a research assistant. You will be given extracted text
from a web page. Summarize the key findings relevant to the research question,
strictly in your own words.

Rules:
- Never copy sentences or distinctive phrases verbatim from the source text.
- Do not exceed 3-4 sentences.
- If the page doesn't actually address the research question, say so briefly
  instead of inventing a summary.
- No markdown, no preamble — just the paraphrased finding.
"""

_SYNTHESIS_SYSTEM_PROMPT = """You are a research assistant producing a short structured
report from several paraphrased findings (already in the researcher's own words, not
quotes from sources). Write a 150-250 word synthesis answering the research question,
organized as 2-4 short paragraphs or a short bulleted list. Do not fabricate claims
beyond what the findings support. No markdown headers, plain prose or simple bullets only.
"""

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass
class SourceFinding:
    source_url: str
    finding: str


def _search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Returns a list of {"url": ..., "content": ...} dicts via the Tavily
    Search API. Requires TAVILY_API_KEY in .env — get a free key (1,000
    searches/month, no card required) at https://app.tavily.com.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.error("research: TAVILY_API_KEY is not set — cannot search. Add it to .env.")
        return []

    try:
        resp = httpx.post(
            _TAVILY_SEARCH_URL,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.error("research: Tavily Search API call failed: %s", exc)
        return []

    results = data.get("results", [])
    return [{"url": r["url"], "content": r.get("content", "")} for r in results if r.get("url")]


def _paraphrase(question: str, raw_text: str) -> str:
    if not raw_text.strip():
        return ""
    user_prompt = f"Research question: {question}\n\nExtracted page text:\n{raw_text}"
    return nim_complete(_PARAPHRASE_SYSTEM_PROMPT, user_prompt, temperature=0.3, max_tokens=200)


def _synthesize(question: str, findings: list[SourceFinding]) -> str:
    if not findings:
        return "No usable sources were found for this question."
    findings_block = "\n".join(f"- ({f.source_url}) {f.finding}" for f in findings)
    user_prompt = f"Research question: {question}\n\nParaphrased findings:\n{findings_block}"
    return nim_complete(_SYNTHESIS_SYSTEM_PROMPT, user_prompt, temperature=0.4, max_tokens=500)


async def run_research(question: str, max_sources: int = 5, headless: bool = True) -> dict:
    # `headless` is accepted for interface compatibility with other agents
    # but unused here — this agent doesn't launch a browser at all.
    results = _search_web(question, max_results=max_sources)
    if not results:
        return {
            "question": question,
            "report": "No usable sources were found for this question "
            "(check that TAVILY_API_KEY is set in .env, or try again — the search API call may have failed).",
            "sources": [],
            "stored_doc_ids": [],
        }

    findings: list[SourceFinding] = []
    for r in results:
        paraphrased = _paraphrase(question, r["content"])
        if paraphrased:
            findings.append(SourceFinding(source_url=r["url"], finding=paraphrased))

    report = _synthesize(question, findings)

    # Store each paraphrased finding as its own embedded row for later
    # semantic search (tools/vector_store.match_research_docs).
    stored_ids = []
    embedding_errors = []
    for f in findings:
        try:
            embedding = nim_embed(f.finding, input_type="passage")
            row = insert_research_doc(query=question, content=f.finding, embedding=embedding, source_url=f.source_url)
            stored_ids.append(row.get("id"))
        except Exception as exc:  # noqa: BLE001 - don't let one bad embed kill the whole run
            logger.error("research: failed to store finding for %s: %s", f.source_url, exc)
            embedding_errors.append({"source_url": f.source_url, "error": str(exc)})

    logger.info("research: question=%r sources_used=%s stored=%s", question, len(findings), len(stored_ids))

    return {
        "question": question,
        "report": report,
        "sources": [f.source_url for f in findings],
        "stored_doc_ids": stored_ids,
        "embedding_errors": embedding_errors,  # empty list if everything stored fine
    }
