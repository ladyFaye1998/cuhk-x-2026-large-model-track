"""Bounded transport for the official Small Track test ZIP.

Public Drive confirmation fields are taken from its actual download form. No
archive body is accepted on HTTP 200. Optional existing HF_TOKEN/Kaggle Secret
access is used only after the public route fails. Secrets and signed URLs are
never returned, logged, or persisted.
"""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import os
import threading
import time
from urllib.parse import urlparse
import weakref

import requests

FILE_ID = "1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4"
ARCHIVE_SIZE = 2_786_949_673
PUBLIC_ACTION = "https://drive.usercontent.google.com/download"
PUBLIC_FORM = "https://drive.google.com/uc"
MIRROR = "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track/resolve/main/Small-Model-Track/Testing/data/small_model_track_test.zip"
MAX_HTML_BYTES = 128 * 1024
MAX_RANGE_BYTES = 16 * 1024 * 1024
MAX_SESSION_BYTES = 32 * 1024 * 1024
_LOCK = threading.RLock()
_STATES = weakref.WeakKeyDictionary()
_HISTORY = []


class _DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active = False
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "form":
            self.active = attrs.get("id") == "download-form"
            if self.active:
                if self.current is not None:
                    raise ValueError("Duplicate download form")
                if attrs.get("action") != PUBLIC_ACTION or attrs.get("method", "get").lower() != "get":
                    raise ValueError("Unexpected official confirmation form action or method")
                self.current = {}
        elif tag == "input" and self.active and attrs.get("type", "").lower() == "hidden":
            name, value = attrs.get("name"), attrs.get("value")
            if not name or value is None or name in self.current:
                raise ValueError("Invalid or duplicate public confirmation field")
            self.current[name] = value

    def handle_endtag(self, tag):
        if tag == "form" and self.active:
            self.forms.append(self.current)
            self.active = False


def parse_confirmation(body):
    parser = _DownloadForm()
    parser.feed(body.decode("utf-8", errors="replace"))
    parser.close()
    if len(parser.forms) != 1:
        raise ValueError("No unique public download confirmation form")
    fields = parser.forms[0]
    if fields.get("id") != FILE_ID or fields.get("export") != "download":
        raise ValueError("Confirmation form does not name the official test archive")
    return fields


def _state(session):
    with _LOCK:
        if session not in _STATES:
            value = {"bytes_received": 0, "html_bytes": 0, "range_requests": 0,
                     "mirror_requests": 0, "confirmation_forms_used": 0,
                     "form": None, "ranges": []}
            _STATES[session] = value
            _HISTORY.append(value)
        return _STATES[session]


def _account(state, amount, *, html=False):
    with _LOCK:
        state["bytes_received"] += amount
        state["html_bytes"] += amount if html else 0
        if state["bytes_received"] > MAX_SESSION_BYTES:
            raise RuntimeError("Official test download exceeded its bounded session budget")


def _bounded_html(response, state):
    kind = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    if kind not in ("text/html", "application/xhtml+xml"):
        raise RuntimeError("Non-range archive response rejected before reading its body")
    length = response.headers.get("Content-Length")
    if length and (not length.isdigit() or int(length) > MAX_HTML_BYTES):
        raise RuntimeError("Oversized confirmation response rejected")
    chunks, received = [], 0
    for block in response.iter_content(4096):
        received += len(block)
        _account(state, len(block), html=True)
        if received > MAX_HTML_BYTES:
            raise RuntimeError("Confirmation HTML exceeded its byte limit")
        chunks.append(block)
    return b"".join(chunks)


def _read_range(response, state, start, end, *, route):
    expected = end - start + 1
    if response.status_code != 206:
        raise RuntimeError("Official range unavailable: HTTP " + str(response.status_code))
    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
        raise RuntimeError("Official response returned a different range or archive size")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise RuntimeError("Encoded range response rejected")
    length = response.headers.get("Content-Length")
    if length and (not length.isdigit() or int(length) != expected):
        raise RuntimeError("Official range Content-Length mismatch")
    chunks, received = [], 0
    for block in response.iter_content(1 << 20):
        received += len(block)
        _account(state, len(block))
        if received > expected:
            raise RuntimeError("Official range exceeded requested byte count")
        chunks.append(block)
    if received != expected:
        raise RuntimeError("Official range was truncated")
    body = b"".join(chunks)
    with _LOCK:
        state["range_requests"] += 1
        state["mirror_requests"] += int(route == "official_hf")
        state["ranges"].append({"start": start, "end_inclusive": end, "bytes": received,
                                "route": route, "sha256": hashlib.sha256(body).hexdigest()})
    return body


def _token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def fetch_range(session, start, end, attempts=8):
    if not (isinstance(start, int) and isinstance(end, int)
            and 0 <= start <= end < ARCHIVE_SIZE and end - start + 1 <= MAX_RANGE_BYTES):
        raise ValueError("Official test range is outside the bounded archive request")
    state = _state(session)
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0"}
    last_error = "range not attempted"
    for attempt in range(min(max(int(attempts), 1), 4)):
        try:
            with _LOCK:
                fields = state["form"]
            if fields is None:
                with session.get(PUBLIC_ACTION, params={"id": FILE_ID, "export": "download", "confirm": "t"},
                                 headers=headers, timeout=(20, 90), stream=True) as response:
                    if response.status_code == 206:
                        return _read_range(response, state, start, end, route="public_drive")
                    if response.status_code == 200:
                        html = _bounded_html(response, state)
                        try:
                            fields = parse_confirmation(html)
                        except ValueError:
                            fields = None
                if fields is None:
                    with session.get(PUBLIC_FORM, params={"export": "download", "id": FILE_ID},
                                     headers={"Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0"},
                                     timeout=(20, 60), stream=True) as response:
                        if response.status_code != 200:
                            raise RuntimeError("Public confirmation unavailable: HTTP " + str(response.status_code))
                        fields = parse_confirmation(_bounded_html(response, state))
                with _LOCK:
                    state["form"] = dict(fields)
                    state["confirmation_forms_used"] += 1
            with session.get(PUBLIC_ACTION, params=fields, headers=headers,
                             timeout=(20, 90), stream=True) as response:
                return _read_range(response, state, start, end, route="public_drive_confirmation")
        except (requests.RequestException, ValueError, RuntimeError) as error:
            # Do not serialize exception text from requests: it can contain a
            # signed redirect URL or request credential details.
            last_error = type(error).__name__
            with _LOCK:
                state["form"] = None
            token = _token()
            if token:
                try:
                    with session.get(MIRROR, headers={**headers, "Authorization": "Bearer " + token},
                                     timeout=(20, 90), stream=True) as response:
                        return _read_range(response, state, start, end, route="official_hf")
                except (requests.RequestException, ValueError, RuntimeError) as mirror_error:
                    last_error = type(mirror_error).__name__
                finally:
                    token = None
            if attempt < min(max(int(attempts), 1), 4) - 1:
                time.sleep(min(8, 2 ** attempt))
    raise RuntimeError("Bounded official test range failed after retries (" + last_error + ")")


def transport_report():
    """Only public identifiers, counters, exact ranges, and content hashes."""
    with _LOCK:
        result = {"file_id": FILE_ID, "archive_size": ARCHIVE_SIZE, "max_session_bytes": MAX_SESSION_BYTES,
                  "sessions": len(_HISTORY)}
        for name in ("bytes_received", "html_bytes", "range_requests", "mirror_requests", "confirmation_forms_used"):
            result[name] = sum(state[name] for state in _HISTORY)
        result["ranges"] = [dict(row) for state in _HISTORY for row in state["ranges"]]
    return result
