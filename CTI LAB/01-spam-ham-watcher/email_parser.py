import email
import re
from email import policy
from html.parser import HTMLParser


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._text = []

    def handle_data(self, data):
        self._text.append(data)

    def get_text(self):
        return "".join(self._text)


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


def extract_email_body(raw_email: bytes) -> str:
    msg = email.message_from_bytes(raw_email, policy=policy.default)
    body = None

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                try:
                    body = part.get_content()
                except Exception:
                    body = part.get_payload(decode=True)
                    if isinstance(body, bytes):
                        body = body.decode("utf-8", errors="replace")
                break
        if body is None:
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/html":
                    try:
                        html = part.get_content()
                    except Exception:
                        html = part.get_payload(decode=True)
                        if isinstance(html, bytes):
                            html = html.decode("utf-8", errors="replace")
                    body = _strip_html(html)
                    break
    else:
        ct = msg.get_content_type()
        try:
            body = msg.get_content()
        except Exception:
            body = msg.get_payload(decode=True)
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="replace")
        if ct == "text/html" and body:
            body = _strip_html(body)

    if body and isinstance(body, str):
        body = re.sub(r"\s+", " ", body).strip()
    return body or ""


def parse_email(raw_email: bytes) -> dict:
    msg = email.message_from_bytes(raw_email, policy=policy.default)
    return {
        "subject": msg.get("subject", ""),
        "from": msg.get("from", ""),
        "to": msg.get("to", ""),
        "date": msg.get("date", ""),
        "body": extract_email_body(raw_email),
    }
