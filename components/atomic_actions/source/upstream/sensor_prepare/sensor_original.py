import csv
import gzip
import io
import json
import math
import re
import struct
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import requests


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")
TRAIN_DISK_IDS = {
    2: "1QbsW4TupA2QMuE8ckph-YT9kP_MKRakV",
    7: "1HW5nxo03oNkw1_Xk1k5Pw6fQH2U_q5XK",
}
TEST_ID = "1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4"
MODALITIES = {"IMU", "Radar", "Skeleton"}
N_BINS = 16
DEVICE_ORDER = ["WTC", "WTRA", "WTLA", "WTRL", "WTLL"]
IMU_CHANNELS = 9
IMU_WIDTH = len(DEVICE_ORDER) * (IMU_CHANNELS * 2 + 1)
POSE_WIDTH = 17 * 3
SKELETON_WIDTH = POSE_WIDTH * 6 + 1
RADAR_WIDTH = 28
SEGMENT_WIDTH = IMU_WIDTH + RADAR_WIDTH + SKELETON_WIDTH
RANGE_CHUNK = 16 * 1024 * 1024


def direct_url(file_id):
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def fetch_range(file_id, start, end, attempts=8):
    expected = end - start + 1
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                direct_url(file_id),
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "Mozilla/5.0",
                },
                timeout=(30, 240),
            )
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != expected:
                raise RuntimeError(
                    f"range status={response.status_code} got={len(response.content)} "
                    f"expected={expected} content_range={response.headers.get('Content-Range')}"
                )
            return response.content
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 1.8 ** attempt))
    raise RuntimeError(f"failed range {file_id} {start}-{end}: {last_error}")


def read_index(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def find_index(pattern, prefer):
    paths = list(INPUT_ROOT.rglob(pattern))
    paths.sort(key=lambda path: (prefer not in str(path), str(path)))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[0]


def modality(name):
    for part in PurePosixPath(name).parts:
        if part in MODALITIES:
            return part
    return None


def is_useful(entry):
    name = PurePosixPath(entry["name"]).name.lower()
    mod = modality(entry["name"])
    if not mod or entry["name"].endswith("/") or name in {".ds_store", "thumbs.db"}:
        return False
    return (mod in {"IMU", "Radar"} and name.endswith(".csv")) or (
        mod == "Skeleton" and name.endswith(".json")
    )


def train_group(name):
    parts = PurePosixPath(name).parts
    mod = modality(name)
    index = parts.index(mod)
    return "/".join(parts[index + 1 : index + 4])


def test_group(name):
    return next(part for part in PurePosixPath(name).parts if part.startswith("SM_test_"))


def numeric_key(value):
    numbers = tuple(int(item) for item in re.findall(r"\d+", str(value)))
    return numbers or (10**9,)


def timestamp_from_name(name):
    match = re.search(
        r"(20\d{2}-\d{2}-\d{2})[_T ](\d{2})[-:](\d{2})[-:](\d{2}(?:\.\d+)?)",
        name,
    )
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)}:{match.group(3)}:{match.group(4)}"


class ContiguousStore:
    def __init__(self, file_id, entries, label):
        self.base = min(entry["offset"] for entry in entries)
        self.end = max(entry["offset"] + entry["compressed"] + 1024 for entry in entries) - 1
        self.path = OUTPUT_ROOT / f"_{label}.bin"
        total = self.end - self.base + 1
        print(f"downloading {label}: {total:,} bytes {self.base}-{self.end}", flush=True)
        with self.path.open("wb") as out:
            for start in range(self.base, self.end + 1, RANGE_CHUNK):
                stop = min(self.end, start + RANGE_CHUNK - 1)
                out.write(fetch_range(file_id, start, stop))
                print(f"  {label}: {stop - self.base + 1:,}/{total:,}", flush=True)
        self.handle = self.path.open("rb")

    def raw(self, entry):
        self.handle.seek(entry["offset"] - self.base)
        header = self.handle.read(30)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {entry['name']}")
        values = struct.unpack("<4s5H3L2H", header)
        name_len, extra_len = values[9], values[10]
        self.handle.seek(name_len + extra_len, 1)
        payload = self.handle.read(entry["compressed"])
        if len(payload) != entry["compressed"]:
            raise RuntimeError(f"short stored payload for {entry['name']}")
        if entry["method"] == 0:
            return payload
        if entry["method"] == 8:
            return zlib.decompress(payload, -15)
        raise RuntimeError(f"unsupported method {entry['method']} for {entry['name']}")

    def close(self):
        self.handle.close()
        self.path.unlink(missing_ok=True)


def merged_intervals(entries):
    rows = sorted(entries, key=lambda entry: entry["offset"])
    merged = []
    for entry in rows:
        start = entry["offset"]
        end = start + entry["compressed"] + 1024 - 1
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end, [entry]])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append(entry)
    return merged


class SparseStore:
    def __init__(self, file_id, entries, workers=8):
        intervals = merged_intervals(entries)
        self.payloads = {}
        total = sum(end - start + 1 for start, end, _ in intervals)
        print(f"downloading test: {len(intervals)} ranges, {total:,} bytes", flush=True)

        def job(item):
            start, end, rows = item
            return start, fetch_range(file_id, start, end), rows

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, item) for item in intervals]
            for future in as_completed(futures):
                start, blob, rows = future.result()
                for entry in rows:
                    rel = entry["offset"] - start
                    header = blob[rel : rel + 30]
                    if header[:4] != b"PK\x03\x04":
                        raise RuntimeError(f"bad test local header for {entry['name']}")
                    values = struct.unpack("<4s5H3L2H", header)
                    name_len, extra_len = values[9], values[10]
                    data_start = rel + 30 + name_len + extra_len
                    payload = blob[data_start : data_start + entry["compressed"]]
                    if len(payload) != entry["compressed"]:
                        raise RuntimeError(f"short test payload for {entry['name']}")
                    if entry["method"] == 0:
                        raw = payload
                    elif entry["method"] == 8:
                        raw = zlib.decompress(payload, -15)
                    else:
                        raise RuntimeError(f"unsupported method {entry['method']}")
                    self.payloads[entry["name"]] = raw
                done += 1
                if done % 50 == 0 or done == len(intervals):
                    print(f"  test ranges: {done}/{len(intervals)}", flush=True)

    def raw(self, entry):
        return self.payloads[entry["name"]]

    def close(self):
        self.payloads.clear()


def decode(raw):
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def split_bins(array, n_bins=N_BINS):
    if len(array) == 0:
        return [array for _ in range(n_bins)]
    indices = np.linspace(0, len(array), n_bins + 1, dtype=int)
    return [array[indices[i] : indices[i + 1]] for i in range(n_bins)]


def parse_imu(store, entries):
    by_device = defaultdict(list)
    for entry in entries:
        try:
            rows = csv.reader(io.StringIO(decode(store.raw(entry))))
            next(rows, None)
            for row in rows:
                if len(row) < 2 + IMU_CHANNELS:
                    continue
                device_raw = row[1].strip().upper()
                device = next((token for token in DEVICE_ORDER if device_raw.startswith(token)), None)
                if device is None:
                    continue
                values = np.asarray(
                    [safe_float(value) for value in row[2 : 2 + IMU_CHANNELS]],
                    dtype=np.float32,
                )
                if np.isfinite(values).sum() >= IMU_CHANNELS // 2:
                    by_device[device].append((row[0], values))
        except Exception:
            continue

    output = np.zeros((N_BINS, IMU_WIDTH), dtype=np.float32)
    for device_index, device in enumerate(DEVICE_ORDER):
        rows = sorted(by_device.get(device, []), key=lambda pair: pair[0])
        base = device_index * (IMU_CHANNELS * 2 + 1)
        if not rows:
            continue
        values = np.stack([row[1] for row in rows])
        medians = np.nanmedian(values, axis=0)
        values = np.where(np.isfinite(values), values, medians)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        for bin_index, chunk in enumerate(split_bins(values)):
            if len(chunk):
                output[bin_index, base : base + IMU_CHANNELS] = chunk.mean(axis=0)
                output[bin_index, base + IMU_CHANNELS : base + 2 * IMU_CHANNELS] = chunk.std(axis=0)
                output[bin_index, base + 2 * IMU_CHANNELS] = np.log1p(len(chunk))
    return output, bool(by_device)


def normalized_pose(points):
    hip = (points[11] + points[12]) / 2.0
    shoulder = (points[5] + points[6]) / 2.0
    scale = float(np.linalg.norm(shoulder - hip))
    if not math.isfinite(scale) or scale < 1e-3:
        scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    scale = max(scale, 1e-3)
    return (points - hip) / scale


def frame_order(name):
    values = tuple(int(value) for value in re.findall(r"\d+", PurePosixPath(name).name))
    return values or (0,)


def parse_skeleton(store, entries):
    frames = []
    for entry in sorted(entries, key=lambda item: frame_order(item["name"])):
        try:
            payload = json.loads(decode(store.raw(entry)))
            objects = payload if isinstance(payload, list) else [payload]
            candidates = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                points = np.asarray(obj.get("keypoints", []), dtype=np.float32)
                if points.shape != (17, 3) or not np.isfinite(points).any():
                    continue
                points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
                scores = np.asarray(obj.get("keypoint_scores", []), dtype=np.float32)
                confidence = float(np.nanmean(scores)) if scores.size else 0.0
                combined = np.concatenate([points.reshape(-1), normalized_pose(points).reshape(-1)])
                candidates.append((confidence, combined))
            if candidates:
                frames.append(max(candidates, key=lambda item: item[0])[1])
        except Exception:
            continue

    output = np.zeros((N_BINS, SKELETON_WIDTH), dtype=np.float32)
    if not frames:
        return output, False
    values = np.stack(frames)
    combined_width = POSE_WIDTH * 2
    for bin_index, chunk in enumerate(split_bins(values)):
        if not len(chunk):
            continue
        output[bin_index, :combined_width] = chunk.mean(axis=0)
        output[bin_index, combined_width : 2 * combined_width] = chunk.std(axis=0)
        if len(chunk) > 1:
            output[bin_index, 2 * combined_width : 3 * combined_width] = np.abs(
                np.diff(chunk, axis=0)
            ).mean(axis=0)
        output[bin_index, -1] = np.log1p(len(chunk))
    return output, True


def parse_radar(store, entries):
    frames = defaultdict(list)
    for entry in entries:
        try:
            reader = csv.reader(io.StringIO(decode(store.raw(entry))))
            header = [cell.strip().lower() for cell in (next(reader, None) or [])]
            wanted = [header.index(token) if token in header else None for token in ("x", "y", "z", "v", "snr", "noise")]
            frame_index = header.index("frame") if "frame" in header else None
            time_index = header.index("timestamp") if "timestamp" in header else None
            for row_number, row in enumerate(reader):
                values = np.asarray(
                    [safe_float(row[index]) if index is not None and index < len(row) else np.nan for index in wanted],
                    dtype=np.float32,
                )
                if not np.isfinite(values).any():
                    continue
                key = row[frame_index] if frame_index is not None and frame_index < len(row) else (
                    row[time_index] if time_index is not None and time_index < len(row) else str(row_number)
                )
                frames[key].append(values)
        except Exception:
            continue

    output = np.zeros((N_BINS, RADAR_WIDTH), dtype=np.float32)
    if not frames:
        return output, False
    frame_vectors = []
    for _, rows in sorted(frames.items(), key=lambda pair: numeric_key(pair[0])):
        values = np.stack(rows)
        medians = np.nanmedian(values, axis=0)
        values = np.where(np.isfinite(values), values, medians)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(values[:, :3], axis=1)
        vector = np.concatenate(
            [
                values.mean(axis=0),
                values.std(axis=0),
                values.min(axis=0),
                values.max(axis=0),
                [np.log1p(len(values)), norms.mean(), norms.std(), float(np.mean(np.abs(values[:, 3])))],
            ]
        )
        frame_vectors.append(vector.astype(np.float32))
    values = np.stack(frame_vectors)
    for bin_index, chunk in enumerate(split_bins(values)):
        if len(chunk):
            output[bin_index] = chunk.mean(axis=0)
    return output, True


def build_split(store, entries, split):
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        group = train_group(entry["name"]) if split == "train" else test_group(entry["name"])
        grouped[group][modality(entry["name"])].append(entry)

    groups = sorted(grouped, key=numeric_key)
    time_bounds = []
    for group in groups:
        timestamps = sorted(
            value
            for rows in grouped[group].values()
            for entry in rows
            for value in [timestamp_from_name(entry["name"])]
            if value is not None
        )
        time_bounds.append((timestamps[0], timestamps[-1]) if timestamps else (None, None))
    segments = np.zeros((len(groups), N_BINS, SEGMENT_WIDTH), dtype=np.float32)
    masks = np.zeros((len(groups), 3), dtype=np.uint8)
    errors = []
    start_time = time.time()
    for group_index, group in enumerate(groups):
        modes = grouped[group]
        try:
            imu, imu_valid = parse_imu(store, modes.get("IMU", []))
            radar, radar_valid = parse_radar(store, modes.get("Radar", []))
            skeleton, skeleton_valid = parse_skeleton(store, modes.get("Skeleton", []))
            segments[group_index] = np.concatenate([imu, radar, skeleton], axis=1)
            masks[group_index] = [imu_valid, radar_valid, skeleton_valid]
        except Exception as exc:
            errors.append({"group": group, "error": repr(exc)})
        if (group_index + 1) % 100 == 0 or group_index + 1 == len(groups):
            print(
                f"{split}: {group_index + 1}/{len(groups)} elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

    global_features = np.concatenate(
        [
            segments.mean(axis=1),
            segments.std(axis=1),
            segments.min(axis=1),
            segments.max(axis=1),
            masks.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return groups, segments, global_features, masks, time_bounds, errors


def main():
    train_index_path = find_index("nonvisual_entries.jsonl.gz", "central-directory")
    test_index_path = find_index("test_nonvisual_entries.jsonl.gz", "selective-nonvisual")
    train_entries = [entry for entry in read_index(train_index_path) if is_useful(entry)]
    test_entries = [entry for entry in read_index(test_index_path) if is_useful(entry)]
    print(f"train entries={len(train_entries):,} test entries={len(test_entries):,}", flush=True)

    train_stores = {}
    for disk in sorted({entry["disk"] for entry in train_entries}):
        rows = [entry for entry in train_entries if entry["disk"] == disk]
        train_stores[disk] = ContiguousStore(TRAIN_DISK_IDS[disk], rows, f"train_disk{disk}")

    class TrainStore:
        def raw(self, entry):
            return train_stores[entry["disk"]].raw(entry)

    test_store = SparseStore(TEST_ID, test_entries)
    train_groups, train_segments, train_global, train_masks, train_times, train_errors = build_split(
        TrainStore(), train_entries, "train"
    )
    test_groups, test_segments, test_global, test_masks, test_times, test_errors = build_split(
        test_store, test_entries, "test"
    )

    train_labels = np.asarray([int(group.split("/", 1)[0].split("_", 1)[0]) for group in train_groups])
    train_users = np.asarray([int(re.search(r"user(\d+)", group).group(1)) for group in train_groups])
    train_trials = np.asarray([group.rsplit("/", 1)[-1] for group in train_groups])
    test_numbers = np.asarray([numeric_key(group)[0] for group in test_groups])

    np.savez_compressed(
        OUTPUT_ROOT / "small_sensor_features.npz",
        train_segments=train_segments,
        train_global=train_global,
        train_masks=train_masks,
        train_labels=train_labels,
        train_users=train_users,
        train_trials=train_trials,
        test_segments=test_segments,
        test_global=test_global,
        test_masks=test_masks,
        test_numbers=test_numbers,
    )
    pd.DataFrame(
        {
            "group": train_groups,
            "label": train_labels,
            "user": train_users,
            "trial": train_trials,
            "imu": train_masks[:, 0],
            "radar": train_masks[:, 1],
            "skeleton": train_masks[:, 2],
            "start_time": [bounds[0] for bounds in train_times],
            "end_time": [bounds[1] for bounds in train_times],
        }
    ).to_csv(OUTPUT_ROOT / "train_sensor_metadata.csv", index=False)
    pd.DataFrame(
        {
            "group": test_groups,
            "test_number": test_numbers,
            "imu": test_masks[:, 0],
            "radar": test_masks[:, 1],
            "skeleton": test_masks[:, 2],
            "start_time": [bounds[0] for bounds in test_times],
            "end_time": [bounds[1] for bounds in test_times],
        }
    ).to_csv(OUTPUT_ROOT / "test_sensor_metadata.csv", index=False)

    manifest = {
        "train_index": str(train_index_path),
        "test_index": str(test_index_path),
        "bins": N_BINS,
        "widths": {
            "imu": IMU_WIDTH,
            "radar": RADAR_WIDTH,
            "skeleton": SKELETON_WIDTH,
            "segment": SEGMENT_WIDTH,
            "global": int(train_global.shape[1]),
        },
        "train": {
            "groups": len(train_groups),
            "segments_shape": list(train_segments.shape),
            "global_shape": list(train_global.shape),
            "labels": dict(Counter(map(str, train_labels))),
            "users": dict(Counter(map(str, train_users))),
            "coverage": train_masks.sum(axis=0).tolist(),
            "nonzero_fraction": float(np.count_nonzero(train_segments) / train_segments.size),
            "errors": train_errors,
        },
        "test": {
            "groups": len(test_groups),
            "segments_shape": list(test_segments.shape),
            "global_shape": list(test_global.shape),
            "coverage": test_masks.sum(axis=0).tolist(),
            "nonzero_fraction": float(np.count_nonzero(test_segments) / test_segments.size),
            "errors": test_errors,
        },
    }
    (OUTPUT_ROOT / "small_sensor_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)

    test_store.close()
    for store in train_stores.values():
        store.close()


if __name__ == "__main__":
    main()
