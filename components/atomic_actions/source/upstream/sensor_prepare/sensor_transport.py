"""Strict byte-range transport for the same official archives used by the vendor.

Adapted from the already verified official_skeleton.py transport. This module has
no pose decoder or learned inputs. Credentials remain in environment/Kaggle
secrets and are never included in receipts, URLs printed to logs or artifacts.
"""
import os
import threading
import time
import requests

FILES = {
    "1HW5nxo03oNkw1_Xk1k5Pw6fQH2U_q5XK": (5368709120, "Small-Model-Track/Training/data/HAR.z08"),
    "1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4": (2786949673, "Small-Model-Track/Testing/data/small_model_track_test.zip"),
}
_LOCAL = threading.local()
_LOCK = threading.Lock()
_MIRROR_IDS = set()
_STATS = {"requests": 0, "downloaded_bytes": 0, "official_mirror_requests": 0}


def token():
    value = os.environ.get("HF_TOKEN")
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def fetch_range(file_id, start, end, attempts=8):
    if file_id not in FILES:
        raise ValueError("Archive is outside this preparation's official source allowlist")
    size, mirror_path = FILES[file_id]
    if not 0 <= start <= end < size:
        raise ValueError("Requested range is outside the official archive")
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    expected = end-start+1
    last_error = "unknown"
    for attempt in range(attempts):
        try:
            mirror = file_id in _MIRROR_IDS
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0"}
            if mirror:
                access = token()
                if not access:
                    raise RuntimeError("Official mirror requires the already authorized HF_TOKEN environment/Kaggle secret")
                url = "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track/resolve/main/" + mirror_path
                headers["Authorization"] = "Bearer " + access
            with _LOCAL.session.get(url, headers=headers, timeout=(20, 120), stream=True) as response:
                if response.status_code != 206:
                    if not mirror:
                        with _LOCK:
                            _MIRROR_IDS.add(file_id)
                        continue
                    raise RuntimeError(f"Official mirror returned HTTP {response.status_code}")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise RuntimeError("Archive range or full size differs from the official index")
                parts, received = [], 0
                for part in response.iter_content(1024*1024):
                    received += len(part)
                    if received > expected:
                        raise RuntimeError("Archive response exceeds its byte budget")
                    parts.append(part)
                if received != expected:
                    raise RuntimeError("Truncated official archive range")
            with _LOCK:
                _STATS["requests"] += 1
                _STATS["downloaded_bytes"] += received
                _STATS["official_mirror_requests"] += int(mirror)
            return b"".join(parts)
        except (requests.RequestException, RuntimeError) as error:
            last_error = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            if attempt+1 < attempts:
                time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"Official archive {file_id} is unavailable: {last_error}")


def receipt():
    return {"method": "Original official archive byte ranges; strict public Drive first, authorized official mirror fallback",
            "source_archives": [{"drive_file_id": fid, "archive_size": size,
                                 "official_mirror_url": "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track/resolve/main/" + path}
                                for fid, (size, path) in FILES.items()],
            "statistics": dict(_STATS), "decoder_changed": False,
            "inherited_transport_source": "official_skeleton.py verified range transport"}
