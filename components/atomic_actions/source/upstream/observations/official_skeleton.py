"""Prepare CUHK-X competition Skeleton observations from public official archives.

Only labeled training Skeleton JSON and unlabeled test Skeleton JSON are read.
The train archive is a multi-part ZIP: its Skeleton local records form one
contiguous span in HAR.z08. Reading self-describing records avoids downloading
the other modalities and does not require the separately hosted ZIP directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import struct
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import requests

TRAIN_PART_ID = "1HW5nxo03oNkw1_Xk1k5Pw6fQH2U_q5XK"
TRAIN_PART_SIZE = 5_368_709_120
TRAIN_START = 3_226_813_822
TRAIN_END = 3_418_728_819  # exclusive; next modality starts here
TRAIN_JSON_COUNT = 86_050
TEST_ID = "1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4"
TEST_SIZE = 2_786_949_673
TEST_JSON_COUNT = 9_201
ALLOWED_TRAIN_USERS = set(range(1, 10)) | set(range(16, 25))
POOL, JOINTS = 128, 17
WORKERS = 8
CHUNK_BYTES = 4 * 1024 * 1024
_LOCAL = threading.local()
_STATS_LOCK = threading.Lock()
_STATS = {"requests": 0, "downloaded_bytes": 0, "official_mirror_requests": 0}
_MIRROR_IDS = set()
_HF_PATHS = {
    TRAIN_PART_ID: "Small-Model-Track/Training/data/HAR.z08",
    TEST_ID: "Small-Model-Track/Testing/data/small_model_track_test.zip",
}


def _hf_token():
    """Use already authorized access; never write credentials to artifacts."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    # An attached private access-token dataset is supported for existing runs.
    # Only this exact filename is read; no file contents are logged or copied.
    root = Path("/kaggle/input")
    if root.exists():
        for path in root.rglob("hf_credentials.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("HF_TOKEN"):
                return value["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def array_sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype, order="C").tobytes()).hexdigest()


def source_url(file_id):
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def fetch_range(file_id, start, end, total_size, attempts=4):
    """Strict bounded HTTP ranges: never read a full archive on HTTP 200."""
    if not (0 <= start <= end < total_size):
        raise ValueError("range is outside the official archive part")
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    expected = end - start + 1
    last_error = "unknown"
    for attempt in range(attempts + 1):
        try:
            mirror = file_id in _MIRROR_IDS
            url = source_url(file_id)
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0"}
            if mirror:
                token = _hf_token()
                if not token:
                    raise RuntimeError("Drive quota exceeded; official mirror requires already authorized HF_TOKEN access")
                url = "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track/resolve/main/" + _HF_PATHS[file_id]
                headers["Authorization"] = "Bearer " + token
            with _LOCAL.session.get(
                url, headers=headers,
                timeout=(20, 120), stream=True,
            ) as response:
                if response.status_code != 206:
                    # Do not download the potentially multi-gigabyte HTTP 200 body.
                    if not mirror:
                        with _STATS_LOCK:
                            _MIRROR_IDS.add(file_id)
                        continue
                    raise RuntimeError(f"official mirror range unavailable: HTTP {response.status_code}")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{total_size}":
                    raise RuntimeError("official archive returned a different byte range or size")
                blocks, received = [], 0
                for block in response.iter_content(1024 * 1024):
                    received += len(block)
                    if received > expected:
                        raise RuntimeError("range exceeded the requested byte budget")
                    blocks.append(block)
                if received != expected:
                    raise RuntimeError("truncated official archive range")
            with _STATS_LOCK:
                _STATS["requests"] += 1
                _STATS["downloaded_bytes"] += received
                _STATS["official_mirror_requests"] += int(mirror)
            return b"".join(blocks)
        except (requests.RequestException, RuntimeError) as error:
            last_error = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            if attempt + 1 < attempts + 1:
                time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"Could not read official archive part {file_id}: {last_error}")


def fetch_span(file_id, start, end, size):
    chunks = [(p, min(p + CHUNK_BYTES, end)) for p in range(start, end, CHUNK_BYTES)]
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_range, file_id, a, b - 1, size): i for i, (a, b) in enumerate(chunks)}
        for number, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if number % 10 == 0 or number == len(chunks):
                print(f"Official archive chunks {number}/{len(chunks)}", flush=True)
    return b"".join(results)


def _zip64(extra, uncomp, comp, offset=0, disk=0):
    pos = 0
    while pos + 4 <= len(extra):
        field, length = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4:pos + 4 + length]
        pos += 4 + length
        if field != 1:
            continue
        values = [uncomp, comp, offset, disk]
        cursor = 0
        for i, sentinel in enumerate((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF)):
            if values[i] == sentinel:
                fmt = "<L" if i == 3 else "<Q"
                values[i] = struct.unpack_from(fmt, body, cursor)[0]
                cursor += struct.calcsize(fmt)
        return values
    return uncomp, comp, offset, disk


def _decode_member(blob, row):
    pos = row["offset"]
    if blob[pos:pos + 4] != b"PK\x03\x04":
        raise RuntimeError("ZIP local header missing")
    v = struct.unpack_from("<4s5H3L2H", blob, pos)
    name_len, extra_len = v[9:11]
    start = pos + 30 + name_len + extra_len
    data = blob[start:start + row["compressed"]]
    if row["method"] == 8:
        data = zlib.decompress(data, -15)
    elif row["method"] != 0:
        raise RuntimeError("unsupported ZIP compression")
    if len(data) != row["uncompressed"] or (zlib.crc32(data) & 0xFFFFFFFF) != row["crc"]:
        raise RuntimeError(f"ZIP integrity check failed: {row['name']}")
    return data


def _local_rows(blob):
    """Read every local record in the fixed modality span, including directories."""
    pos = 0
    while pos < len(blob):
        if blob[pos:pos + 4] != b"PK\x03\x04":
            raise RuntimeError(f"unexpected record at train span offset {pos}")
        v = struct.unpack_from("<4s5H3L2H", blob, pos)
        _, _, flags, method, _, _, crc, comp, uncomp, nl, el = v
        if flags & (1 | 8):
            raise RuntimeError("encrypted or descriptor ZIP record is not supported in the training span")
        name = blob[pos + 30:pos + 30 + nl].decode("utf-8" if flags & 0x800 else "cp437")
        extra = blob[pos + 30 + nl:pos + 30 + nl + el]
        uncomp, comp, _, _ = _zip64(extra, uncomp, comp)
        stop = pos + 30 + nl + el + comp
        if stop > len(blob):
            raise RuntimeError("training span ends within a ZIP record")
        yield {"name": name, "offset": pos, "compressed": comp, "uncompressed": uncomp, "method": method, "crc": crc}
        pos = stop


def _numeric(row):
    return tuple(int(x) for x in re.findall(r"\d+", PurePosixPath(row["name"]).name))


def decode_sequence(rows, read_payload):
    """Largest visible body per frame; keep all XYZ columns; uniform pool if long."""
    frames = []
    for row in sorted(rows, key=_numeric):
        try:
            item = json.loads(read_payload(row).decode("utf-8-sig"))
            people = item if isinstance(item, list) else [item]
            best, spread = None, -1.0
            for person in people:
                if not isinstance(person, dict):
                    continue
                points = np.asarray(person.get("keypoints", []), dtype=np.float32)
                if points.shape != (JOINTS, 3) or not np.isfinite(points).any():
                    continue
                points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
                extent = float(np.linalg.norm(points.max(0) - points.min(0)))
                if extent > spread:
                    best, spread = points, extent
            if best is not None:
                frames.append(best)
        except (UnicodeError, ValueError, TypeError):
            continue
    if not frames:
        raise RuntimeError("official Skeleton clip has no decodable body observations")
    values = np.stack(frames)
    if len(values) > POOL:
        values = values[np.linspace(0, len(values) - 1, POOL).round().astype(np.int64)]
    output = np.zeros((POOL, JOINTS, 3), np.float32)
    output[:len(values)] = values
    return output, len(values)


def prepare_train():
    blob = fetch_span(TRAIN_PART_ID, TRAIN_START, TRAIN_END, TRAIN_PART_SIZE)
    grouped = defaultdict(list)
    count = 0
    for row in _local_rows(blob):
        parts = PurePosixPath(row["name"]).parts
        # Empty folder records can precede the next modality's first file.
        if not row["uncompressed"]:
            continue
        if "Skeleton" not in parts:
            raise RuntimeError("training span includes a different modality")
        if not row["name"].endswith(".json"):
            raise RuntimeError("unexpected non-JSON training Skeleton member")
        index = parts.index("Skeleton")
        action, user, trial = parts[index + 1:index + 4]
        match = re.fullmatch(r"user(\d+)", user)
        if not match or int(match[1]) not in ALLOWED_TRAIN_USERS:
            raise RuntimeError("unexpected subject in official training Skeleton archive")
        if int(action.split("_")[0]) not in range(40):
            raise RuntimeError("unexpected training class")
        grouped["/".join((action, user, trial))].append(row)
        count += 1
    if count != TRAIN_JSON_COUNT or len(grouped) != 2931:
        raise RuntimeError(f"training Skeleton inventory drift: {count} members, {len(grouped)} clips")
    keys = sorted(grouped)
    observations = np.zeros((len(keys), POOL, JOINTS, 3), np.float32)
    counts = np.zeros(len(keys), np.int32)
    metadata = []
    for i, key in enumerate(keys):
        observations[i], counts[i] = decode_sequence(grouped[key], lambda row: _decode_member(blob, row))
        action, user, trial = key.split("/")
        metadata.append(dict(group=key, action=action, user=user, trial=trial, label=int(action.split("_")[0]), n_valid=int(counts[i])))
        if (i + 1) % 500 == 0 or i + 1 == len(keys):
            print(f"Decoded official training clips {i + 1}/{len(keys)}", flush=True)
    return observations, counts, metadata, {"json_members": count, "archive_span_sha256": hashlib.sha256(blob).hexdigest(),
        "archive_part": "HAR.z08", "file_id": TRAIN_PART_ID, "start_inclusive": TRAIN_START, "end_exclusive": TRAIN_END,
        "subject_ids": sorted(ALLOWED_TRAIN_USERS), "source_url": source_url(TRAIN_PART_ID),
        "official_mirror_url": "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track/resolve/main/" + _HF_PATHS[TRAIN_PART_ID]}


def _central_rows(blob):
    pos = 0
    while pos < len(blob):
        if blob[pos:pos + 4] != b"PK\x01\x02":
            raise RuntimeError("test ZIP directory signature missing")
        v = struct.unpack_from("<4s6H3L5H2L", blob, pos)
        flags, method, crc, comp, uncomp = v[3], v[4], v[7], v[8], v[9]
        nl, el, cl, disk, offset = v[10], v[11], v[12], v[13], v[16]
        start = pos + 46
        name = blob[start:start + nl].decode("utf-8" if flags & 0x800 else "cp437")
        extra = blob[start + nl:start + nl + el]
        uncomp, comp, offset, disk = _zip64(extra, uncomp, comp, offset, disk)
        if disk != 0 or flags & 1:
            raise RuntimeError("unexpected split or encrypted test ZIP member")
        yield dict(name=name, offset=offset, compressed=comp, uncompressed=uncomp, method=method, crc=crc)
        pos += 46 + nl + el + cl


def _test_index():
    tail_start = TEST_SIZE - 128 * 1024
    tail = fetch_range(TEST_ID, tail_start, TEST_SIZE - 1, TEST_SIZE)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise RuntimeError("test ZIP end record missing")
    v = struct.unpack_from("<4s4H2LH", tail, pos)
    size, offset = v[5:7]
    if v[4] == 0xFFFF or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        lp = tail.rfind(b"PK\x06\x07", 0, pos)
        if lp < 0:
            raise RuntimeError("test ZIP64 locator missing")
        locator = struct.unpack_from("<4sLQL", tail, lp)
        record = fetch_range(TEST_ID, locator[2], locator[2] + 95, TEST_SIZE)
        if record[:4] != b"PK\x06\x06":
            raise RuntimeError("test ZIP64 end record missing")
        values = struct.unpack_from("<4sQ2H2L4Q", record)
        size, offset = values[8:10]
    central = fetch_span(TEST_ID, offset, offset + size, TEST_SIZE)
    rows = [r for r in _central_rows(central) if "Skeleton" in PurePosixPath(r["name"]).parts and r["name"].endswith(".json")]
    if len(rows) != TEST_JSON_COUNT:
        raise RuntimeError("official test Skeleton entry count drift")
    return rows, hashlib.sha256(central).hexdigest()


def prepare_test():
    rows, directory_sha = _test_index()
    intervals = []
    for row in sorted(rows, key=lambda r: r["offset"]):
        start = row["offset"]
        # Current official local names/extra fields fit below 1,024 bytes.
        stop = min(TEST_SIZE, start + row["compressed"] + 1024)
        if not intervals or start > intervals[-1][1]:
            intervals.append([start, stop, [row]])
        else:
            intervals[-1][1] = max(intervals[-1][1], stop)
            intervals[-1][2].append(row)
    payloads = {}
    def read_interval(item):
        start, stop, members = item
        blob = fetch_range(TEST_ID, start, stop - 1, TEST_SIZE)
        return {r["name"]: _decode_member(blob, dict(r, offset=r["offset"] - start)) for r in members}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(read_interval, interval) for interval in intervals]
        for i, future in enumerate(as_completed(futures), 1):
            payloads.update(future.result())
            if i % 100 == 0 or i == len(futures):
                print(f"Official test Skeleton ranges {i}/{len(futures)}", flush=True)
    groups = defaultdict(list)
    for row in rows:
        key = next(p for p in PurePosixPath(row["name"]).parts if re.fullmatch(r"SM_test_\d{4}", p))
        groups[key].append(row)
    if len(groups) != 405:
        raise RuntimeError("official test clip count drift")
    keys = sorted(groups)
    observations = np.zeros((len(keys), POOL, JOINTS, 3), np.float32)
    counts = np.zeros(len(keys), np.int32)
    metadata = []
    for i, key in enumerate(keys):
        observations[i], counts[i] = decode_sequence(groups[key], lambda r: payloads[r["name"]])
        metadata.append(dict(group=key, path=f"small_model_track_test/{key}/", n_valid=int(counts[i])))
    return observations, counts, metadata, {"json_members": len(rows), "archive_directory_sha256": directory_sha,
        "file_id": TEST_ID, "source_url": source_url(TEST_ID), "range_count": len(intervals)}


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_all(out_dir, template=None):
    """Download observations, write four numeric arrays + metadata; return paths.

    If an official sample is supplied, only its path column is read. Its sample
    prediction values are neither copied nor passed to training.
    """
    started = time.perf_counter()
    before = dict(_STATS)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train, train_counts, train_meta, train_report = prepare_train()
    test, test_counts, test_meta, test_report = prepare_test()
    if template is not None:
        with Path(template).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise RuntimeError("sample submission path column missing")
            paths = [str(row["path"]).replace("\\", "/") for row in reader]
        if len(paths) != len(test_meta) or len(set(paths)) != len(paths):
            raise RuntimeError("sample submission row count or uniqueness mismatch")
        mapping = {row["group"]: i for i, row in enumerate(test_meta)}
        order = [mapping[PurePosixPath(path.rstrip("/")).name] for path in paths]
        if len(set(order)) != len(test_meta):
            raise RuntimeError("sample paths do not cover official test groups exactly")
        test, test_counts = test[order], test_counts[order]
        test_meta = [dict(test_meta[i], path=path) for i, path in zip(order, paths)]
    cache = destination / "skeleton_cache.npz"
    np.savez_compressed(cache, train=train, train_n_valid=train_counts, test=test, test_n_valid=test_counts)
    train_path, test_path = destination / "skeleton_train_meta.csv", destination / "skeleton_test_meta.csv"
    _write_csv(train_path, train_meta, ["group", "action", "user", "trial", "label", "n_valid"])
    _write_csv(test_path, test_meta, ["group", "path", "n_valid"])
    template_path = destination / "sample_submission.csv"
    _write_csv(template_path, [dict(path=r["path"], prediction=0) for r in test_meta], ["path", "prediction"])
    report = {"source": "official competition training and test Skeleton ZIP members", "training": train_report,
        "test": test_report, "labels": "official training action folder; no labels read from test observations",
        "training_clips": len(train), "test_clips": len(test), "pool_frames": POOL, "joints": JOINTS,
        "coordinates": "all three supplied keypoint columns; keypoint_scores and identifiers are not model features",
        "body_selection": "largest Euclidean bounding-box diagonal independently in each frame",
        "empty_clip_policy": "raise an error", "overlong_clip_policy": "128 uniformly spaced valid frames",
        "arrays": {"train": array_sha(train, "<f4"), "train_n_valid": array_sha(train_counts, "<i4"),
            "test": array_sha(test, "<f4"), "test_n_valid": array_sha(test_counts, "<i4")},
        "downloaded_bytes": _STATS["downloaded_bytes"] - before["downloaded_bytes"],
        "official_mirror_requests": _STATS["official_mirror_requests"] - before["official_mirror_requests"],
        "http_requests": _STATS["requests"] - before["requests"], "seconds": time.perf_counter() - started}
    report_path = destination / "data_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(train)} training and {len(test)} test clips in {report['seconds']:.1f}s", flush=True)
    return {"cache": cache, "train_meta": train_path, "test_meta": test_path, "template": template_path, "report": report_path}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--template")
    args = parser.parse_args()
    prepare_all(args.out_dir, args.template)
