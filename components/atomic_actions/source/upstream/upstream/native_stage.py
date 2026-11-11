"""Regenerate native anchors and temporal features from explicit CPU inputs.

Dependencies: Python 3.10+, numpy, pandas, scipy, torch; the three explicit
architecture modules drop.py, DSTformer.py and model_action.py. No downloads,
teacher probability archives, reference CSVs, or historical run reports are read.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import types
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

VIEWS = (1.75, 2.0, 2.25)
ASSIGNMENT_MODE, ASSIGNMENT_WEIGHT = "duplicates_max2", 0.75
MODEL_MODULES = ("drop", "DSTformer", "model_action")


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build_model(model_source_dir, checkpoint):
    source = Path(model_source_dir).resolve()
    # The published module imports model.drop. Keep that import namespace local
    # to construction and restore any host application's existing modules.
    keys = ["model"] + ["model." + name for name in MODEL_MODULES]
    previous = {key: sys.modules.get(key) for key in keys}
    try:
        package = types.ModuleType("model")
        package.__path__ = [str(source)]
        sys.modules["model"] = package
        modules = {}
        for name in MODEL_MODULES:
            spec = importlib.util.spec_from_file_location("model." + name, source / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        backbone = modules["DSTformer"].DSTformer(
            dim_in=3, dim_out=3, dim_feat=64, dim_rep=64, depth=1,
            num_heads=2, mlp_ratio=2, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            maxlen=1, num_joints=17)
        model = modules["model_action"].ActionNet(
            backbone, dim_rep=64, num_classes=40, dropout_ratio=.5,
            version="class", hidden_dim=256, num_joints=17)
        archive = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in archive["model"].items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        # The published constructor ignores hidden_dim and instantiates 2048.
        if model.head.fc1.out_features != 2048 or sum(p.numel() for p in model.parameters()) != 2456365:
            raise ValueError("Unexpected native checkpoint architecture")
        return model
    finally:
        for key, prior in previous.items():
            if prior is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prior


def crop_scale(frames, ratio):
    result = frames.copy()
    valid = frames[:, :, 2] != 0
    count = valid.sum(1)
    xmin = np.min(np.where(valid, frames[:, :, 0], np.inf), axis=1)
    xmax = np.max(np.where(valid, frames[:, :, 0], -np.inf), axis=1)
    ymin = np.min(np.where(valid, frames[:, :, 1], np.inf), axis=1)
    ymax = np.max(np.where(valid, frames[:, :, 1], -np.inf), axis=1)
    scale = np.maximum(xmax - xmin, ymax - ymin) * np.float32(ratio)
    good = (count >= 4) & np.isfinite(scale) & (scale > 0)
    safe = np.where(good, scale, 1).astype(np.float32)
    ox = ((xmin + xmax - safe) * .5).astype(np.float32)
    oy = ((ymin + ymax - safe) * .5).astype(np.float32)
    result[:, :, 0] = ((frames[:, :, 0] - ox[:, None]) / safe[:, None] - .5) * 2
    result[:, :, 1] = ((frames[:, :, 1] - oy[:, None]) / safe[:, None] - .5) * 2
    result = np.clip(result, -1, 1)
    result[~good] = 0
    return np.ascontiguousarray(result, dtype=np.float32)


def softmax(values):
    shifted = values.astype(np.float64) - values.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(1, keepdims=True)


def extract(model, data, counts, batch=512):
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    frames = np.concatenate([row[:int(n)] for row, n in zip(data, counts)])
    rng = np.random.default_rng(1730)
    buckets = rng.integers(0, 256, size=2048, dtype=np.int64)
    signs = rng.choice(np.asarray([-1, 1], np.float32), size=2048)
    projector = csr_matrix((signs * np.float32(np.sqrt(256 / 2048)),
                            (np.arange(2048), buckets)), shape=(2048, 256))
    hidden_sum = np.zeros((len(frames), 256), np.float32)
    logits = []
    started = time.monotonic()
    with torch.inference_mode():
        for ratio in VIEWS:
            prepared = crop_scale(frames, ratio)
            per_view = np.empty((len(frames), 40), np.float32)
            for start in range(0, len(frames), batch):
                stop = min(start + batch, len(frames))
                x = torch.from_numpy(prepared[start:stop, None])
                z = model.backbone.get_representation(x).mean(1).reshape(len(x), -1)
                h = model.head.relu(model.head.bn(model.head.fc1(z)))
                per_view[start:stop] = model.head.fc2(h).numpy()
                hidden_sum[start:stop] += np.asarray(projector.T.dot(h.numpy().T).T, np.float32)
            logits.append(per_view)
            print(json.dumps({"scale_complete": ratio, "frames": len(frames),
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    hidden_sum /= np.float32(len(VIEWS))
    frame_probability = softmax(np.mean(np.stack(logits), axis=0))
    probability = np.stack([frame_probability[a:b].mean(0) for a, b in zip(offsets[:-1], offsets[1:])])
    probability /= probability.sum(1, keepdims=True)
    return probability, hidden_sum, offsets


def normalize(p):
    p = np.clip(np.asarray(p, np.float64), 1e-12, None)
    return p / p.sum(1, keepdims=True)


def batches(meta, train):
    frame = meta.assign(_row=np.arange(len(meta)))
    return [part._row.to_numpy(np.int64) for _, part in frame.groupby(
        ["user", "start_time"] if train else ["start_time"], sort=True, dropna=False)]


def assignment(p, groups, mode=ASSIGNMENT_MODE):
    out = p.copy()
    for rows in groups:
        if len(rows) <= 1:
            continue
        labels = p[rows].argmax(1)
        if mode != "all" and len(np.unique(labels)) == len(rows):
            continue
        r, c = linear_sum_assignment(-np.log(np.clip(p[rows], 1e-12, 1.)))
        assigned = np.empty(len(rows), np.int64)
        assigned[r] = c
        if mode == "duplicates_max2" and np.sum(assigned != labels) > 2:
            continue
        onehot = np.zeros((len(rows), 40), np.float64)
        onehot[np.arange(len(rows)), assigned] = 1.
        out[rows] = onehot
    return out


def temporal_features(hidden, offsets, anchor, counts):
    rows = []
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        z = hidden[a:b].astype(np.float64)
        delta = np.abs(np.diff(z, axis=0)) if len(z) > 1 else np.zeros((1, z.shape[1]), np.float64)
        pooled = [z.mean(0), z.std(0), z.max(0)]
        for bins in (2, 3):
            pooled.extend(z[part].mean(0) if len(part) else np.zeros(z.shape[1], np.float32)
                          for part in np.array_split(np.arange(len(z)), bins))
        pooled.extend([delta.mean(0), delta.std(0), anchor[i], np.array([np.log1p(counts[i])])])
        rows.append(np.concatenate(pooled))
    raw = np.asarray(rows, np.float32)
    rng = np.random.default_rng(1729)
    buckets = rng.integers(0, 512, size=raw.shape[1], dtype=np.int64)
    signs = rng.choice(np.asarray([-1., 1.], np.float32), size=raw.shape[1])
    projected = np.zeros((len(raw), 512), np.float32)
    for col, (bucket, sign) in enumerate(zip(buckets.tolist(), signs.tolist())):
        projected[:, bucket] += np.float32(sign) * raw[:, col]
    projected *= np.float32(np.sqrt(512 / raw.shape[1]))
    return projected


def _metadata(path, required):
    meta = pd.read_csv(path)
    if not set(required).issubset(meta.columns):
        raise ValueError("Missing required observation metadata fields: " + str(path))
    if meta.group.isna().any() or meta.group.astype(str).duplicated().any():
        raise ValueError("Empty or duplicate observation row ID")
    meta["group"] = meta.group.astype(str)
    return meta


def _user_ids(values):
    return np.asarray([int(str(value).removeprefix("user")) for value in values], np.int64)


def _row_order(cache_meta, canonical):
    if set(cache_meta.group) != set(canonical.group):
        raise ValueError("Observation and canonical metadata row IDs differ")
    return pd.Series(np.arange(len(cache_meta)), index=cache_meta.group).loc[canonical.group].to_numpy(np.int64)


def run_stage(skeleton_cache, skeleton_train_meta, skeleton_test_meta, train_meta,
              test_meta, native_checkpoint, model_source_dir, output_dir, *, threads=2, batch=512):
    """Write teacher_arrays.npz and stage_manifest.json in canonical metadata order.

    Cache rows follow skeleton_{train,test}_meta; final rows follow train_meta and
    test_meta. Only official training metadata supplies labels and subject IDs.
    ``start_time`` is the observation acquisition metadata used by the preserved
    fixed batch operation. Test labels are neither required nor accessed.
    """
    started = time.monotonic()
    if not 1 <= threads <= 4 or batch < 1:
        raise ValueError("Use 1--4 CPU threads and a positive batch size")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target, report_path = out / "teacher_arrays.npz", out / "stage_manifest.json"
    if target.exists() or report_path.exists():
        raise FileExistsError("Use a fresh output directory; existing stage artifacts are preserved")
    if shutil.disk_usage(out).free < 50_000_000:
        raise RuntimeError("At least 50 MB free storage is required")
    paths = [(skeleton_cache, "official_skeleton_observations"),
             (skeleton_train_meta, "skeleton_train_row_alignment"),
             (skeleton_test_meta, "skeleton_test_row_alignment"),
             (train_meta, "canonical_training_observation_metadata"),
             (test_meta, "canonical_test_observation_metadata"),
             (native_checkpoint, "native_checkpoint")]
    inputs = [{"path": str(Path(path).resolve()), "sha256": file_sha256(path), "role": role}
              for path, role in paths]
    sources = [Path(__file__).resolve()] + [Path(model_source_dir).resolve() / (name + ".py") for name in MODEL_MODULES]
    source_files = [{"path": str(path), "sha256": file_sha256(path)} for path in sources]
    skeleton_meta = {"train": _metadata(skeleton_train_meta, ("group", "n_valid", "label", "user")),
                     "test": _metadata(skeleton_test_meta, ("group", "n_valid"))}
    canonical = {"train": _metadata(train_meta, ("group", "start_time", "label", "user")),
                 "test": _metadata(test_meta, ("group", "start_time"))}
    order = {split: _row_order(skeleton_meta[split], canonical[split]) for split in ("train", "test")}
    labels = canonical["train"].label.to_numpy(np.int64)
    users = _user_ids(canonical["train"].user)
    if not np.array_equal(labels, skeleton_meta["train"].label.to_numpy(np.int64)[order["train"]]):
        raise ValueError("Training labels disagree across observation metadata")
    if not np.array_equal(users, _user_ids(skeleton_meta["train"].user)[order["train"]]):
        raise ValueError("Training subjects disagree across observation metadata")
    if set(labels) != set(range(40)) or not set(users).issubset(set(range(1, 10)) | set(range(16, 25))):
        raise ValueError("Unexpected official training classes or subjects")
    torch.set_num_threads(threads)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    result = {"labels": labels, "users": users}
    split_receipts = {}
    with threadpool_limits(limits=threads), np.load(skeleton_cache, allow_pickle=False) as archive:
        if set(archive.files) != {"train", "test", "train_n_valid", "test_n_valid"}:
            raise ValueError("Expected an observation-only skeleton NPZ with four arrays")
        model = build_model(model_source_dir, native_checkpoint)
        for split in ("train", "test"):
            data, counts = archive[split], archive[split + "_n_valid"]
            if (data.dtype != np.float32 or data.ndim != 4 or data.shape[2:] != (17, 3)
                    or len(data) != len(skeleton_meta[split]) or not np.isfinite(data).all()
                    or counts.shape != (len(data),) or counts.dtype.kind not in "iu"
                    or np.any(counts < 1) or np.any(counts > data.shape[1])
                    or not np.array_equal(counts, skeleton_meta[split].n_valid.to_numpy())):
                raise ValueError("Invalid skeleton arrays or frame counts for " + split)
            probability, hidden, offsets = extract(model, data, counts, batch=batch)
            raw_probability_hash, hidden_hash = array_sha256(probability), array_sha256(hidden)
            probability = normalize(probability[order[split]])
            groups = batches(canonical[split], split == "train")
            if any(len(rows) > 40 for rows in groups):
                raise ValueError("Acquisition group exceeds the fixed 40-class assignment")
            anchor = normalize((1 - ASSIGNMENT_WEIGHT) * probability + ASSIGNMENT_WEIGHT * assignment(probability, groups))
            inverse = np.argsort(order[split])
            features = temporal_features(hidden, offsets, anchor[inverse], counts)[order[split]]
            result[split + "_anchor"] = anchor
            result[split + "_features"] = features
            result[split + "_groups"] = canonical[split].group.to_numpy(dtype=str)
            split_receipts[split] = {"rows": len(data), "frames": int(counts.sum()),
                "raw_native_probability_sha256": raw_probability_hash,
                "frame_hidden_projection_sha256": hidden_hash, "acquisition_batches": len(groups)}
            del data, counts, probability, hidden, offsets, features, anchor
            gc.collect()
        del model
    for name, array in result.items():
        if array.dtype.kind != "U" and not np.isfinite(array).all():
            raise ValueError("Non-finite generated array: " + name)
    temporary = target.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result)
    if temporary.stat().st_size > 100_000_000:
        raise RuntimeError("Stage output exceeds the 100 MB limit")
    temporary.replace(target)
    manifest = {"stage": "native", "array_file": target.name, "array_sha256": file_sha256(target),
        "array_bytes": target.stat().st_size, "inputs": inputs, "source_files": source_files,
        "learned_inputs_used": [inputs[-1]],
        "method": {"crop_scales": list(VIEWS), "scale_logit_mean_then_frame_softmax_then_clip_mean": True,
            "native_hidden_projection": {"input": 2048, "output": 256, "seed": 1730},
            "batch_assignment": {"mode": ASSIGNMENT_MODE, "weight": ASSIGNMENT_WEIGHT,
                                 "train_keys": ["user", "start_time"], "test_keys": ["start_time"]},
            "temporal_pooling": "mean/std/max, two/three temporal bins, absolute-delta mean/std, anchor, log frame count",
            "temporal_projection": {"output": 512, "seed": 1729}, "training_or_search_performed": False},
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)}
                   for name, value in result.items()}, "splits": split_receipts,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "torch", "threadpoolctl")},
                    "cpu_threads": threads, "frame_batch": batch, "gpu_used": False},
        "seconds": time.monotonic() - started,
        "technical_scope": "Native checkpoint forward inference over decoded Skeleton observations."}
    report_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "native", "output": str(target), "bytes": manifest["array_bytes"],
                      "seconds": manifest["seconds"], "array_sha256": manifest["array_sha256"]}), flush=True)
    return target, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("skeleton-cache", "skeleton-train-meta", "skeleton-test-meta", "train-meta", "test-meta",
                 "native-checkpoint", "model-source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch", type=int, default=512)
    run_stage(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
