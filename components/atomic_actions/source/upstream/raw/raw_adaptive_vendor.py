"""Original raw-adaptive model, preprocessing and XLA training functions.

The training math is preserved. Portable changes: lazy XLA initialization,
selected OOF-state retention, and removal of the original notebook entry point.
"""
import gzip
import json
import math
import os



import random
import re
import struct
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
xm = None  # Initialized explicitly by raw_adaptive_stage, never during import.
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")
N_CLASSES = 40
N_FRAMES = 96
N_JOINTS = 17
N_CHANNELS = 8
EPOCHS = 44
BATCH_SIZE = 96
EVAL_BATCH_SIZE = 128
VALIDATE_FROM = 12
VALIDATE_EVERY = 3
PATIENCE = 5
MAX_MODEL_BYTES = 100_000_000
DEVICE = None
FOLDS = (
    (1, 2, 16, 17),
    (3, 4, 18, 19),
    (5, 6, 20, 21),
    (7, 22, 23),
    (8, 9, 24),
)
TRAIN_DISK_IDS = {
    2: "1QbsW4TupA2QMuE8ckph-YT9kP_MKRakV",
    7: "1HW5nxo03oNkw1_Xk1k5Pw6fQH2U_q5XK",
}
TEST_ID = "1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4"
RANGE_CHUNK = 16 * 1024 * 1024
BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
PARENTS = np.arange(N_JOINTS, dtype=np.int64)
for _parent, _child in BONES:
    if _child != 12 or PARENTS[_child] == _child:
        PARENTS[_child] = _parent
FLIP_PERMUTATION = np.arange(N_JOINTS, dtype=np.int64)
for _left, _right in FLIP_PAIRS:
    FLIP_PERMUTATION[_left], FLIP_PERMUTATION[_right] = (
        FLIP_PERMUTATION[_right],
        FLIP_PERMUTATION[_left],
    )
FLIP_SIGN = np.ones(N_CHANNELS, dtype=np.float32)
FLIP_SIGN[[0, 4]] = -1.0


def find_one(pattern, prefer=None):
    paths = sorted(INPUT_ROOT.rglob(pattern))
    if prefer:
        paths.sort(key=lambda path: (prefer.casefold() not in str(path).casefold(), str(path)))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[0]


def direct_url(file_id):
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def fetch_range(file_id, start, end, attempts=8):
    expected = end - start + 1
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                direct_url(file_id),
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                timeout=(30, 240),
            )
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != expected:
                raise RuntimeError(
                    f"range status={response.status_code} got={len(response.content)} expected={expected}"
                )
            return response.content
        except Exception as error:
            last_error = error
            time.sleep(min(30.0, 1.8 ** attempt))
    raise RuntimeError(f"range {start}-{end} failed: {last_error}")


def read_index(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def is_skeleton(entry):
    parts = PurePosixPath(entry["name"]).parts
    return "Skeleton" in parts and entry["name"].endswith(".json")


def numeric_key(value):
    values = tuple(int(item) for item in re.findall(r"\d+", str(value)))
    return values or (10**9,)


def train_group(name):
    parts = PurePosixPath(name).parts
    index = parts.index("Skeleton")
    return "/".join(parts[index + 1:index + 4])


def test_group(name):
    return next(item for item in PurePosixPath(name).parts if item.startswith("SM_test_"))


def decode_text(raw):
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


class ContiguousStore:
    def __init__(self, file_id, entries, label):
        self.base = min(row["offset"] for row in entries)
        self.end = max(row["offset"] + row["compressed"] + 1024 for row in entries) - 1
        self.path = OUTPUT_ROOT / f"_{label}.bin"
        total = self.end - self.base + 1
        print(f"downloading {label}: {total:,} bytes", flush=True)
        with self.path.open("wb") as output:
            for start in range(self.base, self.end + 1, RANGE_CHUNK):
                end = min(self.end, start + RANGE_CHUNK - 1)
                output.write(fetch_range(file_id, start, end))
        self.handle = self.path.open("rb")

    def raw(self, entry):
        self.handle.seek(entry["offset"] - self.base)
        header = self.handle.read(30)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header {entry['name']}")
        values = struct.unpack("<4s5H3L2H", header)
        self.handle.seek(values[9] + values[10], 1)
        payload = self.handle.read(entry["compressed"])
        if entry["method"] == 0:
            return payload
        if entry["method"] == 8:
            return zlib.decompress(payload, -15)
        raise RuntimeError(f"unsupported ZIP method {entry['method']}")

    def close(self):
        self.handle.close()
        self.path.unlink(missing_ok=True)


def merged_intervals(entries):
    output = []
    for entry in sorted(entries, key=lambda row: row["offset"]):
        start = entry["offset"]
        end = start + entry["compressed"] + 1024 - 1
        if not output or start > output[-1][1] + 1:
            output.append([start, end, [entry]])
        else:
            output[-1][1] = max(output[-1][1], end)
            output[-1][2].append(entry)
    return output


class SparseStore:
    def __init__(self, file_id, entries):
        self.payloads = {}
        intervals = merged_intervals(entries)
        print(f"downloading test skeleton: {len(intervals)} ranges", flush=True)

        def load(item):
            start, end, rows = item
            return start, fetch_range(file_id, start, end), rows

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(load, item) for item in intervals]
            for number, future in enumerate(as_completed(futures), 1):
                start, blob, rows = future.result()
                for entry in rows:
                    relative = entry["offset"] - start
                    header = blob[relative:relative + 30]
                    if header[:4] != b"PK\x03\x04":
                        raise RuntimeError(f"bad test local header {entry['name']}")
                    values = struct.unpack("<4s5H3L2H", header)
                    data_start = relative + 30 + values[9] + values[10]
                    payload = blob[data_start:data_start + entry["compressed"]]
                    if entry["method"] == 8:
                        payload = zlib.decompress(payload, -15)
                    elif entry["method"] != 0:
                        raise RuntimeError(f"unsupported ZIP method {entry['method']}")
                    self.payloads[entry["name"]] = payload
                if number % 100 == 0 or number == len(intervals):
                    print(f"test skeleton ranges {number}/{len(intervals)}", flush=True)

    def raw(self, entry):
        return self.payloads[entry["name"]]

    def close(self):
        self.payloads.clear()


def frame_key(entry):
    return numeric_key(PurePosixPath(entry["name"]).name)


def person_candidates(raw):
    payload = json.loads(decode_text(raw))
    people = payload if isinstance(payload, list) else [payload]
    output = []
    for person in people:
        if not isinstance(person, dict):
            continue
        points = np.asarray(person.get("keypoints", []), dtype=np.float32)
        if points.shape != (N_JOINTS, 3) or not np.isfinite(points).any():
            continue
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32)
        if scores.shape != (N_JOINTS,):
            scores = np.ones(N_JOINTS, dtype=np.float32)
        scores = np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0)
        hip = (points[11] + points[12]) / 2.0
        shoulder = (points[5] + points[6]) / 2.0
        scale = max(float(np.linalg.norm(shoulder - hip)), 1e-3)
        output.append((float(scores.mean()), hip, scale, points, scores))
    return output


def decode_sequence(store, entries):
    tracked_points = []
    tracked_scores = []
    previous_hip = None
    previous_scale = None
    for entry in sorted(entries, key=frame_key):
        try:
            choices = person_candidates(store.raw(entry))
        except Exception:
            choices = []
        if not choices:
            continue
        if previous_hip is None:
            selected = max(choices, key=lambda item: item[0])
        else:
            def tracking_cost(item):
                confidence, hip, scale, _, _ = item
                motion = float(np.linalg.norm(hip - previous_hip)) / max(previous_scale, 1e-3)
                scale_change = abs(math.log(max(scale, 1e-3) / max(previous_scale, 1e-3)))
                return motion + 0.35 * scale_change - 0.10 * confidence
            selected = min(choices, key=tracking_cost)
        _, previous_hip, previous_scale, points, scores = selected
        tracked_points.append(points)
        tracked_scores.append(scores)
    if not tracked_points:
        return np.zeros((N_FRAMES, N_JOINTS, N_CHANNELS), dtype=np.float32), False
    points = np.stack(tracked_points).astype(np.float32)
    scores = np.stack(tracked_scores).astype(np.float32)
    selected = np.linspace(0, len(points) - 1, N_FRAMES).round().astype(np.int64)
    points = points[selected]
    scores = scores[selected]
    hips = (points[:, 11] + points[:, 12]) / 2.0
    shoulders = (points[:, 5] + points[:, 6]) / 2.0
    scales = np.linalg.norm(shoulders - hips, axis=1)
    finite_scales = scales[np.isfinite(scales) & (scales > 1e-3)]
    clip_scale = float(np.median(finite_scales)) if len(finite_scales) else 1.0
    clip_scale = max(clip_scale, 1e-3)
    relative = np.clip((points - hips[:, None, :]) / clip_scale, -5.0, 5.0)
    root_motion = np.clip((hips - hips[:1]) / clip_scale, -5.0, 5.0)
    scale_motion = np.clip(scales / clip_scale - 1.0, -2.0, 2.0)
    output = np.zeros((N_FRAMES, N_JOINTS, N_CHANNELS), dtype=np.float32)
    output[..., :3] = relative
    output[..., 3] = scores
    output[..., 4:7] = root_motion[:, None, :]
    output[..., 7] = scale_motion[:, None]
    return output, True


def build_split(store, entries, split):
    grouped = defaultdict(list)
    for entry in entries:
        group = train_group(entry["name"]) if split == "train" else test_group(entry["name"])
        grouped[group].append(entry)
    groups = sorted(grouped, key=numeric_key)
    values = np.zeros((len(groups), N_FRAMES, N_JOINTS, N_CHANNELS), dtype=np.float32)
    valid = np.zeros(len(groups), dtype=np.float32)
    for index, group in enumerate(groups):
        values[index], present = decode_sequence(store, grouped[group])
        valid[index] = float(present)
        if (index + 1) % 250 == 0 or index + 1 == len(groups):
            print(f"{split} sequences {index + 1}/{len(groups)}", flush=True)
    return groups, values, valid


def adjacency():
    matrix = np.eye(N_JOINTS, dtype=np.float32)
    for left, right in BONES:
        matrix[left, right] = 1.0
        matrix[right, left] = 1.0
    matrix /= matrix.sum(axis=1, keepdims=True)
    return torch.from_numpy(matrix)


class SkeletonDataset(Dataset):
    def __init__(self, values, valid, labels, users, rows, augment):
        self.values = values
        self.valid = valid
        self.labels = labels
        self.users = users
        self.rows = np.asarray(rows, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        values = self.values[row].copy()
        if self.augment:
            if np.random.random() < 0.5:
                values[..., 0] *= -1.0
                values[..., 4] *= -1.0
                permutation = np.arange(N_JOINTS)
                for left, right in FLIP_PAIRS:
                    permutation[left], permutation[right] = permutation[right], permutation[left]
                values = values[:, permutation]
            angle = np.random.uniform(-0.14, 0.14)
            cosine, sine = math.cos(angle), math.sin(angle)
            for offset in (0, 4):
                x = values[..., offset].copy()
                y = values[..., offset + 1].copy()
                values[..., offset] = cosine * x - sine * y
                values[..., offset + 1] = sine * x + cosine * y
            values[..., :3] *= np.random.uniform(0.94, 1.06)
            noise = np.random.normal(0.0, 0.012, size=values[..., :3].shape).astype(np.float32)
            values[..., :3] += noise
            dropped = np.random.random(N_FRAMES) < 0.025
            values[dropped] = 0.0
            shift = int(np.random.randint(-4, 5))
            if shift:
                values = np.roll(values, shift, axis=0)
                if shift > 0:
                    values[:shift] = 0.0
                else:
                    values[shift:] = 0.0
        return (
            torch.from_numpy(values),
            torch.tensor([self.valid[row]], dtype=torch.float32),
            torch.tensor(self.labels[row], dtype=torch.long),
            torch.tensor(self.users[row], dtype=torch.long),
        )


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        assert out_channels % groups == 0
        relation_channels = max(8, in_channels // 4)
        self.groups = groups
        self.value = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.theta = nn.Conv1d(in_channels, relation_channels, 1, bias=False)
        self.phi = nn.Conv1d(in_channels, relation_channels, 1, bias=False)
        self.relation = nn.Conv2d(relation_channels, groups, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.10))
        self.register_buffer("base_graph", adjacency())
        self.norm = nn.GroupNorm(8, out_channels)

    def forward(self, values):
        batch, _, frames, joints = values.shape
        pooled = values.mean(dim=2)
        relation = torch.tanh(self.theta(pooled).unsqueeze(-1) - self.phi(pooled).unsqueeze(-2))
        dynamic = torch.softmax(self.relation(relation), dim=-1)
        graph = self.base_graph[None, None] + self.alpha.abs() * dynamic
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        projected = self.value(values)
        channels = projected.shape[1] // self.groups
        projected = projected.reshape(batch, self.groups, channels, frames, joints)
        projected = torch.einsum("bgctv,bgvw->bgctw", projected, graph)
        projected = projected.reshape(batch, self.groups * channels, frames, joints)
        return self.norm(projected)


class AdaptiveGraphTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.10):
        super().__init__()
        self.graph = AdaptiveGraphConv(in_channels, out_channels)
        self.temporal = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (5, 1),
                stride=(stride, 1),
                padding=(2, 0),
                groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, values):
        return F.gelu(self.temporal(self.graph(values)) + self.skip(values))


class StreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            AdaptiveGraphTemporalBlock(N_CHANNELS, 64, dropout=0.06),
            AdaptiveGraphTemporalBlock(64, 96, stride=2, dropout=0.08),
            AdaptiveGraphTemporalBlock(96, 128, dropout=0.10),
            AdaptiveGraphTemporalBlock(128, 160, stride=2, dropout=0.12),
        )
        self.attention = nn.Sequential(nn.Linear(160, 80), nn.Tanh(), nn.Linear(80, 1))

    def forward(self, values):
        values = self.blocks(values.permute(0, 3, 1, 2))
        values = values.mean(dim=3).transpose(1, 2)
        attention = self.attention(values).softmax(dim=1)
        return torch.cat(
            [(values * attention).sum(dim=1), values.mean(dim=1), values.amax(dim=1)],
            dim=1,
        )


class AdaptiveTwoStreamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_stream = StreamEncoder()
        self.bone_stream = StreamEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(160 * 6 + 1, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.26),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.classifier = nn.Linear(256, N_CLASSES)
        self.projector = nn.Sequential(nn.Linear(256, 192), nn.GELU(), nn.Linear(192, 128))
        self.register_buffer("parents", torch.from_numpy(PARENTS.copy()))
        self.register_buffer("flip_permutation", torch.from_numpy(FLIP_PERMUTATION.copy()))
        self.register_buffer("flip_sign", torch.from_numpy(FLIP_SIGN.copy()))

    def flip_values(self, values):
        return (
            values.index_select(2, self.flip_permutation)
            * self.flip_sign.view(1, 1, 1, N_CHANNELS)
        )

    def bone_values(self, joints):
        parents = self.parents
        parent_values = joints[:, :, parents]
        # Keep this construction functional.  In-place slice updates on a
        # derived XLA tensor can trigger functionalization/version-counter
        # failures, especially when the same forward is used under no_grad for
        # validation.  These three pieces exactly preserve the intended bone
        # channels: relative XYZ, conservative confidence, and clip-level root
        # motion/scale copied from the joint stream.
        return torch.cat(
            [
                joints[..., :3] - parent_values[..., :3],
                torch.minimum(joints[..., 3:4], parent_values[..., 3:4]),
                joints[..., 4:8],
            ],
            dim=-1,
        )

    def forward(self, joints, valid, return_projection=True):
        bones = self.bone_values(joints)
        features = torch.cat(
            [self.joint_stream(joints), self.bone_stream(bones), valid], dim=1
        )
        embedding = self.fusion(features)
        logits = self.classifier(embedding)
        projection = F.normalize(self.projector(embedding), dim=1) if return_projection else None
        return logits, projection


def subject_invariant_supcon(features, labels, users, temperature=0.10):
    similarity = features @ features.T / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & users[:, None].ne(users[None, :]) & ~identity
    denominator_mask = ~identity
    exp_similarity = torch.exp(similarity) * denominator_mask
    log_probability = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_counts = positives.sum(dim=1)
    valid = positive_counts > 0
    loss = -(log_probability * positives).sum(dim=1) / positive_counts.clamp_min(1)
    valid_weights = valid.to(loss.dtype)
    return (loss * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)


def predict_logits(model, values, valid, rows, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.zeros((0, N_CLASSES), dtype=np.float32)
    device_outputs = []
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            selected = rows[begin:begin + batch_size]
            if len(selected) < batch_size:
                selected = np.concatenate(
                    [selected, np.full(batch_size - len(selected), selected[-1], dtype=np.int64)]
                )
            batch = values[selected].to(DEVICE)
            available = valid[selected].to(DEVICE)
            logits, _ = model(batch, available, return_projection=False)
            flipped_logits, _ = model(
                model.flip_values(batch), available, return_projection=False
            )
            device_outputs.append((logits + flipped_logits) * 0.5)
            # Materialize one fixed-shape inference graph per batch while retaining
            # its small logits tensor on device. Host transfer happens only once.
            xm.mark_step()
        result = torch.cat(device_outputs, dim=0)[:len(rows)].cpu().numpy()
    assert result.shape == (len(rows), N_CLASSES)
    return result.astype(np.float32, copy=False)


def load_cpu_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_xla_state(model, path):
    # xm.save performs a coordinated device-to-host conversion instead of one
    # synchronous .cpu() call per parameter.
    xm.mark_step()
    xm.save(model.state_dict(), str(path))
    xm.wait_device_ops()


def train_model(values, valid, labels, users, train_rows, valid_rows, seed, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        xm.set_rng_state(seed, DEVICE)
    except TypeError:
        xm.set_rng_state(seed)
    dataset = SkeletonDataset(values.numpy(), valid.numpy().ravel(), labels, users, train_rows, True)
    counts = np.bincount(labels[train_rows], minlength=N_CLASSES).astype(np.float64)
    sample_weights = 1.0 / np.sqrt(np.maximum(counts[labels[train_rows]], 1.0))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            len(train_rows),
            replacement=True,
            generator=torch.Generator().manual_seed(seed + 91),
        ),
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    model = AdaptiveTwoStreamNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.035)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=1.5e-3,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=3e-5
    )
    best_path = OUTPUT_ROOT / f"_adaptive_xla_best_{seed}.pt"
    best_path.unlink(missing_ok=True)
    best_epoch, best_score, stale = 0, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch, available, target, subject in loader:
            batch = batch.to(DEVICE)
            available = available.to(DEVICE)
            target = target.to(DEVICE)
            subject = subject.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits, projection = model(batch, available)
            loss = criterion(logits, target)
            loss = loss + 0.08 * subject_invariant_supcon(projection, target, subject)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0, foreach=False)
            xm.optimizer_step(optimizer, barrier=False)
            xm.mark_step()
        scheduler.step()
        should_validate = (
            valid_rows is not None
            and epoch >= VALIDATE_FROM
            and epoch % VALIDATE_EVERY == 0
        )
        if should_validate:
            logits = predict_logits(model, values, valid, valid_rows)
            score = float(np.mean(logits.argmax(1) == labels[valid_rows]))
            if score > best_score:
                save_xla_state(model, best_path)
                best_epoch, best_score, stale = epoch, score, 0
            else:
                stale += 1
            print(
                json.dumps(
                    {
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "validation_accuracy": score,
                        "best_accuracy": float(best_score),
                        "stale_checks": int(stale),
                    }
                ),
                flush=True,
            )
            if stale >= PATIENCE:
                break
    if not best_path.exists():
        save_xla_state(model, best_path)
        best_epoch = epochs
    best_state = load_cpu_state(best_path)
    model.load_state_dict(best_state)
    xm.mark_step()
    best_path.unlink(missing_ok=True)
    returned_state = best_state  # Retain selected OOF weights as well as full-fit weights.
    return model, returned_state, best_epoch, best_score


ORIGINAL_SOURCE_SHA256 = "0892aa0dc7f6d4f48da31f6058a4b032254cd7f38ad11d7dc29b4741e4fef652"
