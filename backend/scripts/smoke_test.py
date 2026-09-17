"""
End-to-end smoke test for PRTECH Business OS.

Run this against a live server (uvicorn must already be running) to check
that every agent and endpoint is actually working — not just that the code
compiles, but that the real external calls (Groq, NVIDIA NIM, Tavily,
Supabase, OpenStreetMap) all succeed right now. This project's free-tier
providers have repeatedly retired models or hit transient network issues
mid-development, so this exists to turn "did that break anything?" into a
30-second check instead of manually re-running curl commands for every
agent.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --base-url http://localhost:8000
    python scripts/smoke_test.py --skip lead_gen,monitor   # skip slow/networked ones

Exit code is 0 if everything passed, 1 if anything failed — safe to use in
a CI step or a pre-push check, not just interactively.

Notes on what "pass" means per test:
- Most tests just check for a 200 status and a response shape that looks
  right (e.g. "agent_output" present, no unexpected "error" key). They
  don't guarantee semantic correctness (e.g. that lead_gen actually found
  real leads) — OSM/Tavily coverage varies by query, so a test asserting
  "found > 0" would be flaky for reasons that have nothing to do with code
  correctness. Where that trade-off matters, it's called out in the test.
- form_fill and monitor tests hit public test-friendly targets
  (httpbin.org) rather than requiring you to supply real ones.
- lead_gen/monitor/form_fill are the slowest (real browser/network work);
  skip them with --skip for a fast check of just the LLM-backed agents.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    duration_s: float = 0.0


@dataclass
class SmokeTest:
    base_url: str
    client: httpx.Client = field(init=False)
    results: list[TestResult] = field(default_factory=list)

    def __post_init__(self):
        self.client = httpx.Client(base_url=self.base_url, timeout=60)

    def run(self, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        start = time.time()
        try:
            passed, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a raised exception is itself a failure to report
            passed, detail = False, f"raised {type(exc).__name__}: {exc}"
        duration = time.time() - start
        self.results.append(TestResult(name, passed, detail, duration))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} ({duration:.1f}s){' - ' + detail if detail else ''}")

    def post_chat(self, message: str, params: dict | None = None) -> dict:
        resp = self.client.post("/chat", json={"message": message, "params": params or {}})
        resp.raise_for_status()
        return resp.json()

    # --- individual tests ---------------------------------------------

    def test_health(self) -> tuple[bool, str]:
        resp = self.client.get("/health")
        return resp.status_code == 200, f"status={resp.status_code}"

    def test_router_lead_gen(self) -> tuple[bool, str]:
        data = self.post_chat("find dentists in Coimbatore")
        ok = data.get("intent") == "lead_gen" and "error" not in (data.get("agent_output") or {})
        return ok, json.dumps(data.get("agent_output"))[:200]

    def test_router_research(self) -> tuple[bool, str]:
        data = self.post_chat("research recent trends in local SEO")
        output = data.get("agent_output") or {}
        ok = data.get("intent") == "research" and bool(output.get("report")) and not output.get("embedding_errors")
        return ok, f"sources={len(output.get('sources', []))} embedding_errors={output.get('embedding_errors')}"

    def test_router_outreach(self) -> tuple[bool, str]:
        data = self.post_chat("outreach to dentists in Coimbatore")
        output = data.get("agent_output") or {}
        # draft_only is the safety-critical assertion here — a regression
        # that silently flips this to auto_send would be a real incident.
        ok = data.get("intent") == "outreach" and output.get("mode") == "draft_only"
        return ok, f"mode={output.get('mode')} drafted={output.get('drafted')}"

    def test_router_social(self) -> tuple[bool, str]:
        data = self.post_chat("announce our new offer")
        output = data.get("agent_output") or {}
        ok = data.get("intent") == "social" and output.get("mode") == "draft_only" and output.get("post")
        return ok, f"mode={output.get('mode')}"

    def test_form_fill(self) -> tuple[bool, str]:
        data = self.post_chat(
            "fill the test form",
            params={
                "form_url": "https://httpbin.org/forms/post",
                "rows": [{"custname": "Smoke Test", "custemail": "smoke@example.com"}],
                "field_selectors": {"custname": "input[name=custname]", "custemail": "input[name=custemail]"},
                "dry_run": True,
            },
        )
        output = data.get("agent_output") or {}
        ok = data.get("intent") == "form_fill" and output.get("mode") == "dry_run" and output.get("succeeded", 0) > 0
        return ok, f"succeeded={output.get('succeeded')} failed={output.get('failed')}"

    def test_monitor(self) -> tuple[bool, str]:
        resp = self.client.post("/monitor/add", json={"url": "https://httpbin.org/uuid"})
        resp.raise_for_status()
        data = resp.json()
        ok = "changed" in data and "current_hash" in data
        return ok, f"changed={data.get('changed')} first_check={data.get('first_check')}"

    def test_audit_log(self) -> tuple[bool, str]:
        resp = self.client.get("/audit-log", params={"limit": 5})
        resp.raise_for_status()
        rows = resp.json()
        ok = isinstance(rows, list) and len(rows) > 0
        return ok, f"rows_returned={len(rows) if isinstance(rows, list) else 'n/a'}"

    def test_clarify_fallback(self) -> tuple[bool, str]:
        data = self.post_chat("asdkfjasldkfj nonsense unrelated text")
        # Not asserting on the exact intent — the classifier may reasonably
        # route ambiguous nonsense to "clarify" or occasionally guess an
        # intent; what matters is the server doesn't 500.
        return "intent" in data, f"intent={data.get('intent')}"


_ALL_TESTS = [
    ("health", SmokeTest.test_health),
    ("router+lead_gen", SmokeTest.test_router_lead_gen),
    ("router+research", SmokeTest.test_router_research),
    ("router+outreach (draft-only check)", SmokeTest.test_router_outreach),
    ("router+social (draft-only check)", SmokeTest.test_router_social),
    ("form_fill (dry-run)", SmokeTest.test_form_fill),
    ("monitor", SmokeTest.test_monitor),
    ("audit-log", SmokeTest.test_audit_log),
    ("clarify fallback (no 500 on nonsense)", SmokeTest.test_clarify_fallback),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated substrings to skip, matched against test names, e.g. --skip lead_gen,monitor",
    )
    args = parser.parse_args()

    skip_terms = [s.strip() for s in args.skip.split(",") if s.strip()]

    print(f"PRTECH Business OS — smoke test against {args.base_url}\n")

    tester = SmokeTest(base_url=args.base_url)
    for name, method in _ALL_TESTS:
        if any(term in name for term in skip_terms):
            print(f"  [SKIP] {name}")
            continue
        tester.run(name, lambda m=method: m(tester))

    passed = sum(1 for r in tester.results if r.passed)
    total = len(tester.results)
    print(f"\n{passed}/{total} tests passed.")

    if passed < total:
        print("\nFailed tests:")
        for r in tester.results:
            if not r.passed:
                print(f"  - {r.name}: {r.detail}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
