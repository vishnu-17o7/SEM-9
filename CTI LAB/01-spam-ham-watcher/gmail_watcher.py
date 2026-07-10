#!/usr/bin/env python3
import imaplib
import logging
import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from email_parser import extract_email_body
from spam_utils import classify

from db import init_db, record_classification

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
MODEL_PATH = os.getenv("MODEL_PATH", "spam_text_model.joblib")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

_running = True


def _setup_logging():
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def _signal_handler(signum, frame):
    global _running
    log.info("Shutdown signal received, stopping watcher...")
    _running = False


def _to_text(eid) -> str:
    """imaplib returns ids as bytes on older Python, str on 3.9+. Normalise."""
    if isinstance(eid, bytes):
        return eid.decode("ascii", errors="replace")
    return str(eid)


def _find_spam_folder(mail) -> str | None:
    candidates = ["[Gmail]/Spam", "[Google Mail]/Spam", "Spam", "Junk"]
    for name in candidates:
        try:
            typ, _ = mail.select(name, readonly=True)
        except Exception:
            continue
        if typ == "OK":
            try:
                mail.close()
            except Exception:
                pass
            return name
    try:
        typ, folders = mail.list()
    except Exception:
        return None
    if typ != "OK":
        return None
    for line in folders:
        decoded = line.decode(errors="replace") if isinstance(line, bytes) else str(line)
        for marker in ("Spam", "spam", "Junk", "junk"):
            if marker in decoded:
                parts = decoded.split('"')
                folders_in_quotes = [p for p in parts if p.strip() and "/" not in p and marker.lower() in p.lower()]
                for p in folders_in_quotes:
                    try:
                        typ2, _ = mail.select(p, readonly=True)
                    except Exception:
                        continue
                    if typ2 == "OK":
                        try:
                            mail.close()
                        except Exception:
                            pass
                        return p
    return None


def _ensure_config():
    errors = []
    if not EMAIL_ADDRESS:
        errors.append("EMAIL_ADDRESS is not set in .env file")
    if not EMAIL_PASSWORD:
        errors.append("EMAIL_PASSWORD is not set in .env file")
    if not Path(MODEL_PATH).exists():
        errors.append(f"Model file '{MODEL_PATH}' not found. Run train_text_model.py first.")
    if errors:
        for e in errors:
            log.error(e)
        log.error("Copy .env.template to .env and fill in your credentials.")
        return False
    return True


def main():
    global log
    log = _setup_logging()

    if not _ensure_config():
        sys.exit(1)

    try:
        init_db()
    except Exception as e:
        log.error(f"Failed to initialize classification DB: {e}")
        sys.exit(1)
    log.info("Recording classifications to DB")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log.info(f"Starting Gmail watcher for {EMAIL_ADDRESS}")
    log.info(f"Polling every {POLL_INTERVAL}s, model: {MODEL_PATH}, batch_size={BATCH_SIZE}")

    while _running:
        try:
            _check_inbox()
        except Exception as e:
            log.error(f"Error during check: {e}")

        if _running:
            log.debug(f"Sleeping for {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)

    log.info("Watcher stopped.")


def _check_inbox():
    log.debug("Connecting to IMAP server...")
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("INBOX")

    typ, data = mail.search(None, "UNSEEN")
    if typ != "OK":
        log.warning("Failed to search INBOX")
        mail.logout()
        return

    email_ids = data[0].split()
    if not email_ids:
        log.debug("No new emails.")
        mail.logout()
        return

    if len(email_ids) > BATCH_SIZE:
        log.info(
            f"Found {len(email_ids)} new email(s); processing first {BATCH_SIZE} this poll"
        )
        email_ids = email_ids[:BATCH_SIZE]
    else:
        log.info(f"Found {len(email_ids)} new email(s)")

    spam_folder = _find_spam_folder(mail)
    try:
        mail.select("INBOX")
    except Exception as e:
        log.warning(f"Could not re-select INBOX after folder scan: {e}")
        mail.logout()
        return
    spam_count = 0

    import email as _email

    for eid in email_ids:
        if not _running:
            break

        eid_str = _to_text(eid)
        try:
            typ, data = mail.fetch(eid, "(RFC822)")
            if typ != "OK" or not data or not data[0]:
                log.warning(
                    f"FETCH failed for email {eid_str} (typ={typ}); "
                    "aborting this poll"
                )
                break

            raw_email = data[0][1]
            body = extract_email_body(raw_email)
            if not body:
                log.debug(f"Skipping email {eid_str}: no extractable body")
                mail.store(eid, "+FLAGS", "\\Seen")
                continue

            is_spam, prob = classify(body, path=MODEL_PATH)
            label = "SPAM" if is_spam else "ham"

            msg = mail.fetch(eid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
            subject = "(unknown)"
            sender = "(unknown)"
            if msg and msg[0]:
                hdr_bytes = None
                for part in msg[1] if len(msg) > 1 else []:
                    if isinstance(part, tuple) and len(part) >= 2:
                        hdr_bytes = part[1]
                        break
                if hdr_bytes is not None:
                    if isinstance(hdr_bytes, bytes):
                        hdr = _email.message_from_bytes(hdr_bytes)
                    else:
                        hdr = _email.message_from_string(str(hdr_bytes))
                    subject = hdr.get("Subject", "(unknown)") or "(unknown)"
                    sender = hdr.get("From", "(unknown)") or "(unknown)"

            log.info(
                f"[{label.upper():>4}] ({prob:.1%}) From: {sender}  Subject: {subject}"
            )

            try:
                record_classification(
                    sender=sender,
                    subject=subject,
                    label="spam" if is_spam else "ham",
                    probability=prob,
                    body_preview=(body or "")[:200],
                )
            except Exception as e:
                log.warning(f"DB write failed: {e}")

            if is_spam and spam_folder:
                try:
                    mail.copy(eid, spam_folder)
                    mail.store(eid, "+FLAGS", "\\Deleted")
                    spam_count += 1
                except Exception as e:
                    log.warning(f"Failed to move spam email {eid_str}: {e}")
            else:
                try:
                    mail.store(eid, "+FLAGS", "\\Seen")
                except Exception as e:
                    log.warning(f"Failed to mark email {eid_str} as seen: {e}")

        except Exception as e:
            log.warning(f"Error processing email {eid_str}: {e}")

    if spam_count > 0:
        try:
            mail.expunge()
            log.info(f"Moved {spam_count} spam email(s) to {spam_folder}")
        except Exception as e:
            log.warning(f"Expunge failed: {e}")

    try:
        mail.close()
    except Exception:
        pass
    try:
        mail.logout()
    except Exception:
        pass


if __name__ == "__main__":
    main()
