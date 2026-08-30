"""
Social Poster Agent

Input: a content brief, e.g. "announce our new whitening treatment offer".
Behavior:
  1. Drafts platform-appropriate post copy via NVIDIA NIM.
  2. Drafts reply suggestions for a batch of incoming comments/DMs, if any
     are supplied.
  3. In draft-only mode (the default): returns drafts for human review,
     posts/replies nothing.
  4. In auto-post mode (opt-in via auto_post=True): calls a pluggable
     `poster` backend to actually publish. No backend is wired in by
     default — see _NotConfiguredPoster below.

IMPORTANT — platform ToS:
Posting or DM'ing via browser automation instead of official platform APIs
generally violates Instagram's and LinkedIn's Terms of Service, and can get
the automating account suspended or banned. This agent is deliberately
built around a `SocialPoster` interface rather than a browser-automation
implementation, so you plug in the official API for whichever platform
you're using instead:
  - Instagram: Instagram Graph API (requires a Business/Creator account
    linked to a Facebook Page, and Meta app review for some permissions)
  - LinkedIn: LinkedIn Marketing API / Share on LinkedIn API (requires app
    review for most posting scopes)
Both are free to use (no per-post cost) but require going through each
platform's developer registration — there's no way around that step
without violating ToS. If you want a browser-automation prototype anyway
(e.g. for a personal account, understanding the ban risk), you can
implement SocialPoster using tools/browser.py + tools/browser_runner.py
the same way lead_gen.py does, but that's a conscious risk trade-off you'd
be opting into, not something this module defaults to.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tools.llm import nim_complete

logger = logging.getLogger("prtech.agents.social_poster")

_POST_SYSTEM_PROMPT = """You are a social media copywriter for a local business. Write a short,
platform-appropriate post based on the content brief given.

Rules:
- Keep it under 280 characters for a punchy, platform-agnostic post.
- No excessive hashtags (2-4 max, relevant only).
- No hype-heavy language or multiple exclamation points.
- Output ONLY the post text. No preamble, no markdown, no quotes around it.
"""

_REPLY_SYSTEM_PROMPT = """You are drafting a reply to a comment or DM on a business's social
media post. Write a short, friendly, on-brand reply.

Rules:
- Keep it under 2 sentences.
- Match a warm, human tone — not corporate or scripted-sounding.
- If the comment is a complaint or negative, acknowledge it and suggest moving
  to a private channel (DM or phone) rather than resolving it publicly.
- Output ONLY the reply text. No preamble, no markdown.
"""


@dataclass
class DraftPost:
    platform: str
    content: str


@dataclass
class DraftReply:
    platform: str
    original_comment: str
    reply: str


class SocialPoster(ABC):
    """
    Pluggable backend for actually publishing. Implement this against the
    official Instagram Graph API / LinkedIn API for your platform — see the
    module docstring for why browser automation isn't the default here.
    """

    @abstractmethod
    async def publish_post(self, platform: str, content: str) -> bool:
        ...

    @abstractmethod
    async def publish_reply(self, platform: str, comment_id: str, reply: str) -> bool:
        ...


class _NotConfiguredPoster(SocialPoster):
    async def publish_post(self, platform: str, content: str) -> bool:
        raise NotImplementedError(
            f"No SocialPoster backend is configured for auto-posting to {platform}. "
            "Implement the SocialPoster interface against the official Instagram Graph API "
            "or LinkedIn API and pass it as `poster=` to run_social_poster."
        )

    async def publish_reply(self, platform: str, comment_id: str, reply: str) -> bool:
        raise NotImplementedError(
            f"No SocialPoster backend is configured for auto-replying on {platform}. "
            "Implement the SocialPoster interface against the official Instagram Graph API "
            "or LinkedIn API and pass it as `poster=` to run_social_poster."
        )


def _draft_post(platform: str, brief: str) -> str:
    user_prompt = f"Platform: {platform}\nContent brief: {brief}\n\nWrite the post now."
    return nim_complete(_POST_SYSTEM_PROMPT, user_prompt, temperature=0.6, max_tokens=150)


def _draft_reply(platform: str, comment: str) -> str:
    user_prompt = f"Platform: {platform}\nIncoming comment/DM: {comment}\n\nWrite the reply now."
    return nim_complete(_REPLY_SYSTEM_PROMPT, user_prompt, temperature=0.5, max_tokens=100)


async def run_social_poster(
    brief: str | None = None,
    platform: str = "instagram",
    incoming_comments: list[dict] | None = None,
    auto_post: bool = False,
    poster: SocialPoster | None = None,
) -> dict:
    """
    incoming_comments: optional list of {"id": ..., "text": ...} dicts to
    draft replies for, e.g. pulled from the platform's API separately.
    """
    poster = poster or _NotConfiguredPoster()
    result = {"mode": "auto_post" if auto_post else "draft_only", "post": None, "replies": []}

    if brief:
        content = _draft_post(platform, brief)
        post_entry = {"platform": platform, "content": content, "status": "draft"}

        if auto_post:
            try:
                ok = await poster.publish_post(platform, content)
                post_entry["status"] = "published" if ok else "failed"
            except NotImplementedError as exc:
                post_entry["status"] = "failed"
                post_entry["error"] = str(exc)
                logger.error("social_poster: %s", exc)

        result["post"] = post_entry

    for comment in incoming_comments or []:
        reply = _draft_reply(platform, comment.get("text", ""))
        reply_entry = {
            "platform": platform,
            "comment_id": comment.get("id"),
            "original_comment": comment.get("text"),
            "reply": reply,
            "status": "draft",
        }

        if auto_post:
            try:
                ok = await poster.publish_reply(platform, comment.get("id", ""), reply)
                reply_entry["status"] = "published" if ok else "failed"
            except NotImplementedError as exc:
                reply_entry["status"] = "failed"
                reply_entry["error"] = str(exc)
                logger.error("social_poster: %s", exc)

        result["replies"].append(reply_entry)

    logger.info(
        "social_poster: platform=%s mode=%s post_drafted=%s replies_drafted=%s",
        platform,
        result["mode"],
        bool(result["post"]),
        len(result["replies"]),
    )

    return result
