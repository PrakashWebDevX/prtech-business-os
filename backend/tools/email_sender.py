"""
Minimal SMTP sender. Works with Gmail (via an App Password) or any other
standard SMTP provider — set SMTP_HOST/PORT/USER/PASSWORD in .env.

Kept deliberately simple (no attachments, plain + optional HTML body).
Swap in the Gmail API instead if you need OAuth-based sending or higher
volume than Gmail's SMTP limits allow.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger("prtech.tools.email_sender")


def send_email(to_addr: str, subject: str, body_text: str, body_html: str | None = None) -> bool:
    """Returns True on success, False on failure (never raises to the caller)."""
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
        return True
    except Exception as exc:  # noqa: BLE001 - log and report failure, don't crash the agent
        logger.error("send_email failed for %s: %s", to_addr, exc)
        return False
