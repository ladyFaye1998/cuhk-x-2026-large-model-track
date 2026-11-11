"""Strict official-archive ranges with public Drive confirmation and HF fallback.

Only the four archive parts needed for nonvisual preparation are allowed. No
credentials, confirmation parameters or signed redirect URLs are persisted.
"""
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import threading
import time

import requests

FILES = {
    "1QbsW4TupA2QMuE8ckph-YT9kP_MKRakV": (5368709120, "Small-Model-Track/Training/data/HAR.z03"),
    "1HW5nxo03oNkw1_Xk1k5Pw6fQH2U_q5XK": (5368709120, "Small-Model-Track/Training/data/HAR.z08"),
    "1nlJIpcRuL2FRcapJaFVS3eC-uZuedtqh": (1673136305, "Small-Model-Track/Training/data/HAR.zip"),
    "1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4": (2786949673, "Small-Model-Track/Testing/data/small_model_track_test.zip"),
}
_LOCAL = threading.local()
_LOCK = threading.Lock()
_FORMS = {}
_MIRRORS = set()
_RECEIPTS = []
_PRIVATE_ACCESS_MOUNTS = (
    Path('/kaggle/input/cuhkx-hf-credentials-private'),
    Path('/kaggle/input/datasets/ladyfaye/cuhkx-hf-credentials-private'),
)


class ConfirmationForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.action = None
        self.fields = {}
        self.form_count = 0
        self.method = None
        self.duplicate_fields = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.form_count += 1
            self.action = attrs.get("action")
            self.method = attrs.get("method", "get").lower()
            self.active = True
        if self.active and tag == "input" and attrs.get("type", "").lower() == "hidden" and "name" in attrs:
            if attrs["name"] in self.fields:
                self.duplicate_fields = True
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def _attached_private_token():
    """Read only the existing account attachment, in memory; never serialize it."""
    matches = {}
    try:
        for directory in _PRIVATE_ACCESS_MOUNTS:
            candidate = directory / 'hf_credentials.json'
            if not candidate.exists():
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError('Unexpected access file type')
            resolved = candidate.resolve(strict=True)
            if resolved.parent != directory.resolve(strict=True):
                raise ValueError('Access file escaped its exact attachment directory')
            matches[resolved] = candidate
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError('Ambiguous access attachment')
        candidate = next(iter(matches.values()))
        if not 1 <= candidate.stat().st_size <= 4096:
            raise ValueError('Access attachment size is invalid')
        payload = candidate.read_bytes()
        if len(payload) > 4096:
            raise ValueError('Access attachment exceeded its bound')
        value = json.loads(payload.decode('utf-8'))
        token = value.get('HF_TOKEN') if isinstance(value, dict) else None
        if not isinstance(token, str) or not 1 <= len(token) <= 2048 or any(not 33 <= ord(c) <= 126 for c in token):
            raise ValueError('Invalid access value')
        return token
    except (OSError, ValueError, TypeError):
        # Parsing/path exceptions can contain input fragments. Never forward
        # their text or exception chain to notebook logs or receipts.
        raise RuntimeError('The existing private access attachment could not be read safely') from None


def _token():
    value = os.environ.get("HF_TOKEN")
    if value:
        return value
    if not hasattr(_LOCAL, "token"):
        try:
            from kaggle_secrets import UserSecretsClient
            _LOCAL.token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            _LOCAL.token = None
        if not _LOCAL.token:
            _LOCAL.token = _attached_private_token()
    return _LOCAL.token


def _get_range_response(file_id, start, end, mirror):
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    session = _LOCAL.session
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0"}
    if mirror:
        token = _token()
        if not token:
            raise RuntimeError("Official mirror requires existing environment, Kaggle Secret or private attachment access")
        headers["Authorization"] = "Bearer " + token
        return session.get("https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track/resolve/main/" + FILES[file_id][1], headers=headers, stream=True, timeout=(20,120))
    cached = _FORMS.get(file_id)
    if cached:
        return session.get(cached[0], params=cached[1], headers=headers, stream=True, timeout=(20,120))
    response = session.get("https://drive.google.com/uc", params={"export":"download", "id":file_id}, headers=headers, stream=True, timeout=(20,120))
    if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", "").lower():
        return response
    # Consume only a bounded HTML warning. Never consume an ignored-range archive.
    html, size = [], 0
    for chunk in response.iter_content(8192):
        size += len(chunk)
        if size > 65536:
            response.close()
            raise RuntimeError("Drive confirmation response exceeded its HTML budget")
        html.append(chunk)
    response.close()
    form = ConfirmationForm()
    form.feed(b"".join(html).decode("utf-8", errors="replace"))
    if (form.form_count != 1 or form.method != "get" or form.duplicate_fields
            or form.action != "https://drive.usercontent.google.com/download"
            or form.fields.get("id") != file_id or form.fields.get("export") != "download"):
        raise RuntimeError("Public Drive range unavailable; valid same-file confirmation form absent")
    with _LOCK:
        _FORMS[file_id] = (form.action, form.fields)
    return session.get(form.action, params=form.fields, headers=headers, stream=True, timeout=(20,120))


def fetch_range(file_id, start, end, total_size=None, attempts=4):
    """Read one inclusive range; the optional size must match the fixed archive."""
    if file_id not in FILES:
        raise ValueError("Archive outside the nonvisual source allowlist")
    size, mirror_path = FILES[file_id]
    if total_size is not None and total_size != size:
        raise ValueError("Caller archive size differs from the official source")
    if not 0 <= start <= end < size:
        raise ValueError("Range lies outside the archive part")
    expected = end-start+1
    last_error = "unknown"
    for attempt in range(attempts+1):
        mirror = file_id in _MIRRORS
        try:
            with _get_range_response(file_id, start, end, mirror) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"Official range returned HTTP {response.status_code}")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise RuntimeError("Returned archive range/size differs from the request")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise RuntimeError("Archive response uses unexpected content encoding")
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and (not declared_length.isdigit() or int(declared_length) != expected):
                    raise RuntimeError("Archive response declares a different byte count")
                parts, received = [], 0
                for chunk in response.iter_content(1024*1024):
                    received += len(chunk)
                    if received > expected:
                        raise RuntimeError("Archive response exceeds its byte budget")
                    parts.append(chunk)
                if received != expected:
                    raise RuntimeError("Truncated archive range")
            blob = b"".join(parts)
            with _LOCK:
                _RECEIPTS.append({"file_id":file_id,"archive_path":mirror_path,"start":start,"end":end,"bytes":received,
                                  "sha256":hashlib.sha256(blob).hexdigest(),"transport":"official_hf_mirror" if mirror else "public_drive"})
            return blob
        except (requests.RequestException, RuntimeError) as error:
            last_error = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            if not mirror:
                with _LOCK:
                    _MIRRORS.add(file_id)
                continue
            if attempt < attempts:
                time.sleep(min(8,2**attempt))
    raise RuntimeError(f"Official archive {file_id} is unavailable: {last_error}")


def fetch_span(file_id, start, end, total_size=None, chunk_bytes=4*1024*1024):
    """Read a half-open span in bounded sequential chunks."""
    if end <= start:
        raise ValueError("Empty/reversed archive span")
    return b"".join(fetch_range(file_id, p, min(p+chunk_bytes,end)-1, total_size) for p in range(start,end,chunk_bytes))


def receipts():
    with _LOCK:
        return [dict(row) for row in _RECEIPTS]
