"""Original Large Track IMU, skeleton and radar feature parser."""
# Recovered Kaggle source SHA256: d7511883a9db29aa0935957fce27a9bb874cb35d2752ef7f6f2727d7a5807f13
import csv
import io
import json
import math
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
import numpy as np
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working')
N_BINS = 8
DEVICE_ORDER = ['WTC', 'WTRA', 'WTLA', 'WTRL', 'WTLL']
IMU_CHANNELS = 9
IMU_WIDTH = len(DEVICE_ORDER) * (IMU_CHANNELS * 2 + 1)
SKELETON_WIDTH = 17 * 3 * 3 + 1
RADAR_WIDTH = 6 * 2 + 2
SEGMENT_WIDTH = IMU_WIDTH + SKELETON_WIDTH + RADAR_WIDTH

def find_archive():
    candidates = sorted(INPUT_ROOT.rglob('*.zip'))
    preferred = [p for p in candidates if 'sensor' in str(p).lower() or 'lmt' in p.name.lower()]
    if not preferred:
        raise FileNotFoundError(f'No LMT sensor zip under {INPUT_ROOT}: {candidates}')
    return preferred[0]

def decode(raw):
    for encoding in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode('utf-8', errors='replace')

def unit_key(name):
    match = re.search('/Training/HAU/(user\\d+)/([^/]+)/', name, re.I)
    if match:
        return f'HAU/{match.group(1).lower()}/{match.group(2)}'
    match = re.search('/Training/HARn/([^/]+)/(user\\d+)/([^/]+)/', name, re.I)
    if match:
        return f'HARn/{match.group(1)}/{match.group(2).lower()}/{match.group(3)}'
    match = re.search('/Testing/large_model_track_test/(LM_test_\\d+)/', name, re.I)
    if match:
        return f'large_model_track_test/{match.group(1)}'
    return None

def modality(name):
    match = re.search('/(IMU|Radar|Skeleton)/', name, re.I)
    return match.group(1).lower() if match else None

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
    return [array[indices[i]:indices[i + 1]] for i in range(n_bins)]

def parse_imu(zf, infos):
    by_device = defaultdict(list)
    for info in infos:
        if not info.filename.lower().endswith('.csv'):
            continue
        try:
            rows = csv.reader(io.StringIO(decode(zf.read(info))))
            next(rows, None)
            for row in rows:
                if len(row) < 2 + IMU_CHANNELS:
                    continue
                device_raw = row[1].strip().upper()
                device = next((token for token in DEVICE_ORDER if device_raw.startswith(token)), None)
                if device is None:
                    continue
                values = np.asarray([safe_float(v) for v in row[2:2 + IMU_CHANNELS]], dtype=np.float32)
                if np.isfinite(values).sum() < IMU_CHANNELS // 2:
                    continue
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
            if len(chunk) == 0:
                continue
            output[bin_index, base:base + IMU_CHANNELS] = chunk.mean(axis=0)
            output[bin_index, base + IMU_CHANNELS:base + IMU_CHANNELS * 2] = chunk.std(axis=0)
            output[bin_index, base + IMU_CHANNELS * 2] = np.log1p(len(chunk))
    return output

def frame_order(name):
    values = [int(value) for value in re.findall('\\d+', PurePosixPath(name).name)]
    return tuple(values) if values else (0,)

def normalized_pose(obj):
    points = np.asarray(obj.get('keypoints', []), dtype=np.float32)
    if points.shape != (17, 3) or not np.isfinite(points).any():
        return None
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    hip = (points[11] + points[12]) / 2.0
    shoulder = (points[5] + points[6]) / 2.0
    scale = float(np.linalg.norm(shoulder - hip))
    if not math.isfinite(scale) or scale < 0.001:
        scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    scale = max(scale, 0.001)
    return ((points - hip) / scale).reshape(-1)

def parse_skeleton(zf, infos):
    frames = []
    for info in sorted(infos, key=lambda item: frame_order(item.filename)):
        if not info.filename.lower().endswith('.json'):
            continue
        try:
            payload = json.loads(decode(zf.read(info)))
            objects = payload if isinstance(payload, list) else [payload]
            candidates = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                pose = normalized_pose(obj)
                if pose is None:
                    continue
                scores = np.asarray(obj.get('keypoint_scores', []), dtype=np.float32)
                confidence = float(np.nanmean(scores)) if scores.size else 0.0
                candidates.append((confidence, pose))
            if candidates:
                frames.append(max(candidates, key=lambda item: item[0])[1])
        except Exception:
            continue
    output = np.zeros((N_BINS, SKELETON_WIDTH), dtype=np.float32)
    if not frames:
        return output
    values = np.stack(frames)
    for bin_index, chunk in enumerate(split_bins(values)):
        if len(chunk) == 0:
            continue
        width = 17 * 3
        output[bin_index, :width] = chunk.mean(axis=0)
        output[bin_index, width:width * 2] = chunk.std(axis=0)
        if len(chunk) > 1:
            output[bin_index, width * 2:width * 3] = np.abs(np.diff(chunk, axis=0)).mean(axis=0)
        output[bin_index, -1] = np.log1p(len(chunk))
    return output

def parse_radar(zf, infos):
    rows = []
    for info in infos:
        if not info.filename.lower().endswith('.csv'):
            continue
        try:
            reader = csv.reader(io.StringIO(decode(zf.read(info))))
            header = [cell.strip().lower() for cell in next(reader, None) or []]
            wanted = []
            for token in ('x', 'y', 'z', 'v', 'snr', 'noise'):
                try:
                    wanted.append(header.index(token))
                except ValueError:
                    wanted.append(None)
            for row in reader:
                values = np.asarray([safe_float(row[index]) if index is not None and index < len(row) else np.nan for index in wanted], dtype=np.float32)
                if np.isfinite(values).any():
                    rows.append(values)
        except Exception:
            continue
    output = np.zeros((N_BINS, RADAR_WIDTH), dtype=np.float32)
    if not rows:
        return output
    values = np.stack(rows)
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, medians)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    for bin_index, chunk in enumerate(split_bins(values)):
        if len(chunk) == 0:
            continue
        output[bin_index, :6] = chunk.mean(axis=0)
        output[bin_index, 6:12] = chunk.std(axis=0)
        output[bin_index, 12] = np.log1p(len(chunk))
        output[bin_index, 13] = float(np.mean(np.linalg.norm(chunk[:, :3], axis=1)))
    return output
