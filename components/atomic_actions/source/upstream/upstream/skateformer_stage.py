"""Portable SkateFormer forward inference from observations and fold states.

Dependencies: numpy, pandas, torch, torchvision, timm, safetensors, threadpoolctl.
The vendor directory contains the unchanged published architecture and recovered
deterministic preprocessing. This stage performs no fine-tuning or downloads.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset

from skateformer_vendor import SkateFormer_official as architecture
from skateformer_vendor import skateformer_locked_preprocessing_v1 as adapter

FOLDS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (16, 17, 18), (19, 20, 21), (22, 23, 24))
HELD_ROWS = (477, 435, 530, 530, 478, 481)
MODEL_CONFIG = dict(num_classes=40, num_people=2, num_frames=64, num_points=24,
    kernel_size=7, num_heads=32, type_1_size=(8, 8), type_2_size=(8, 12),
    type_3_size=(8, 8), type_4_size=(8, 12), attn_drop=.5, head_drop=0.,
    drop=0., rel=True, drop_path=.2, mlp_ratio=4., index_t=True, global_pool="avg")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def normalize(value):
    value = np.asarray(value, np.float64)
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value.sum(1) <= 0):
        raise ValueError("Invalid model probabilities")
    return value / value.sum(1, keepdims=True)


class ObservationDataset(Dataset):
    def __init__(self, data, counts, rows, modality):
        if modality not in ("joint", "bone"):
            raise ValueError(modality)
        self.data, self.counts = data, counts
        self.rows, self.modality = np.asarray(rows, np.int64), modality

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        sequence = self.data[row, :int(self.counts[row])]
        pseudo = adapter.map_h36m17_to_ntu25(sequence)
        raw = adapter.source_native_translate_and_duplicate(pseudo)
        clip, index_t = adapter.crop_resize_eval(raw)
        if self.modality == "bone":
            clip = adapter.joint_to_bone(clip)
        clip = adapter.partition_joints(clip)
        return torch.from_numpy(np.ascontiguousarray(clip)).float(), torch.from_numpy(np.ascontiguousarray(index_t)).float(), row


def load_model(checkpoint, device="cpu"):
    state = load_file(str(checkpoint["path"]), device="cpu")
    if len(state) != 214:
        raise ValueError("Unexpected SkateFormer checkpoint entry count")
    if any(t.dtype != torch.float16 for t in state.values() if t.is_floating_point()):
        raise ValueError("Expected the historical FP16 replay state")
    canonical_hash = canonical_state_sha256(state)
    if checkpoint.get("canonical_state_sha256") and checkpoint["canonical_state_sha256"] != canonical_hash:
        raise ValueError("SkateFormer checkpoint canonical state mismatch")
    model = architecture.SkateFormer_(**MODEL_CONFIG)
    model.load_state_dict(state, strict=True)
    model.float().to(device).eval()
    del state
    return model, canonical_hash


def predict(model, data, counts, rows, modality, batch=32):
    device = next(model.parameters()).device
    loader = DataLoader(ObservationDataset(data, counts, rows, modality), batch_size=batch,
                        shuffle=False, drop_last=False, num_workers=0)
    probabilities, actual = [], []
    with torch.inference_mode():
        for features, index_t, row_indices in loader:
            logits = model(features.to(device, non_blocking=True), index_t.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            actual.append(row_indices.numpy())
    if not np.array_equal(np.concatenate(actual), np.asarray(rows, np.int64)):
        raise ValueError("SkateFormer prediction row order changed")
    value = np.concatenate(probabilities).astype(np.float32, copy=False)
    if value.shape != (len(rows), 40) or not np.isfinite(value).all():
        raise ValueError("SkateFormer probability shape or finiteness mismatch")
    return value


def _metadata(path, required):
    frame = pd.read_csv(path)
    if not set(required).issubset(frame.columns) or frame.group.isna().any() or frame.group.duplicated().any():
        raise ValueError("Invalid observation metadata: " + str(path))
    frame["group"] = frame.group.astype(str).str.replace("\\", "/", regex=False)
    return frame


def _canonical_order(source, target):
    if set(source.group) != set(target.group):
        raise ValueError("Canonical and skeleton row IDs differ")
    return pd.Series(np.arange(len(source)), index=source.group).loc[target.group].to_numpy(np.int64)


def _load_checkpoint_map(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(spec.get("folds", [])) != 6:
        raise ValueError("Checkpoint map must contain six folds")
    paths = []
    for fold, record in enumerate(spec["folds"]):
        if record["fold"] != fold or tuple(record["held_users"]) != FOLDS[fold]:
            raise ValueError("Checkpoint fold mapping changed")
        for modality in ("joint", "bone"):
            item = record[modality]
            item["path"] = str((Path(path).resolve().parent / item["path"]).resolve())
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Checkpoint file fingerprint mismatch")
            paths.append(dict(item, role="skateformer_fold_checkpoint", fold=fold, modality=modality))
    return spec, paths


def run_stage(skeleton_cache, skeleton_train_meta, skeleton_test_meta, train_meta,
              test_meta, checkpoint_map, output_dir, *, threads=2, batch=32,
              smoke_samples=0, maximum_full_seconds=600, device="cpu"):
    """Regenerate six-fold OOF train_skate and twelve-model mean test_skate.

    Fold0 uses the backbone-transfer pair, whose replay states must be bound by
    the accompanying equivalence receipt to the semantic experiment's control.
    The separate semantic treatment pair is never a replacement for that control.
    """
    started = time.monotonic()
    if not 1 <= threads <= 2 or batch < 1 or not 0 <= smoke_samples <= 16:
        raise ValueError("Use at most two CPU threads; smoke permits at most 16 clips per modality")
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose device cpu or cuda explicitly")
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise ValueError("CUDA requires a deterministic CUBLAS_WORKSPACE_CONFIG")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no automatic fallback")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target, manifest_path = out / "teacher_arrays.npz", out / "stage_manifest.json"
    smoke_path = out / (device + "_cost_smoke.json")
    if not smoke_samples and (target.exists() or manifest_path.exists()):
        raise FileExistsError("Existing stage artifacts are preserved; choose a fresh output directory")
    if smoke_samples and smoke_path.exists():
        raise FileExistsError("Existing smoke receipt is preserved; choose a fresh output directory")
    torch.set_num_threads(threads)
    torch.manual_seed(20260823)
    if device == "cuda":
        torch.cuda.manual_seed_all(20260823)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec, checkpoints = _load_checkpoint_map(checkpoint_map)
    input_pairs = [(skeleton_cache, "official_skeleton_observations"),
        (skeleton_train_meta, "skeleton_train_row_alignment"), (skeleton_test_meta, "skeleton_test_row_alignment"),
        (train_meta, "canonical_training_metadata"), (test_meta, "canonical_test_metadata"),
        (checkpoint_map, "checkpoint_mapping")]
    inputs = [{"path": str(Path(p).resolve()), "sha256": sha256(p), "role": role} for p, role in input_pairs]
    inputs.extend(checkpoints)
    vendor = Path(__file__).resolve().parent / "skateformer_vendor"
    source_files = [{"path": str(p), "sha256": sha256(p)} for p in
        [Path(__file__).resolve(), vendor / "SkateFormer_official.py", vendor / "skateformer_locked_preprocessing_v1.py"]]
    sm = {"train": _metadata(skeleton_train_meta, ("group", "user", "label", "n_valid")),
          "test": _metadata(skeleton_test_meta, ("group", "n_valid"))}
    cm = {"train": _metadata(train_meta, ("group", "user", "label")), "test": _metadata(test_meta, ("group",))}
    orders = {split: _canonical_order(sm[split], cm[split]) for split in ("train", "test")}
    users = np.asarray([int(str(v).removeprefix("user")) for v in sm["train"].user], np.int64)
    labels = sm["train"].label.to_numpy(np.int64)
    if (not np.array_equal(labels[orders["train"]], cm["train"].label.to_numpy(np.int64))
            or not np.array_equal(users[orders["train"]], cm["train"].user.to_numpy(np.int64))):
        raise ValueError("Canonical training labels or subjects disagree")
    with np.load(skeleton_cache, allow_pickle=False) as z:
        if set(z.files) != {"train", "test", "train_n_valid", "test_n_valid"}:
            raise ValueError("An observation-only four-array skeleton cache is required")
        data = {split: z[split] for split in ("train", "test")}
        counts = {split: z[split + "_n_valid"] for split in ("train", "test")}
    for split in data:
        if (data[split].dtype != np.float32 or data[split].shape != (len(sm[split]), 128, 17, 3)
                or not np.isfinite(data[split]).all() or np.any(counts[split] < 1) or np.any(counts[split] > 128)
                or not np.array_equal(counts[split], sm[split].n_valid.to_numpy())):
            raise ValueError("Invalid skeleton observation arrays")
    fold_indices = [np.flatnonzero(np.isin(users, held)) for held in FOLDS]
    if tuple(map(len, fold_indices)) != HELD_ROWS or set(np.concatenate(fold_indices)) != set(range(len(users))):
        raise ValueError("Six-fold training subject population changed")
    runtimes, model_receipts = [], []
    with threadpool_limits(limits=threads):
        if smoke_samples:
            rows = fold_indices[0][:smoke_samples]
            for modality in ("joint", "bone"):
                tick = time.monotonic()
                model, state_hash = load_model(spec["folds"][0][modality], device=device)
                inference_start = time.monotonic()
                p = predict(model, data["train"], counts["train"], rows, modality, batch=batch)
                runtime = time.monotonic() - inference_start
                runtimes.append(runtime)
                model_receipts.append({"modality": modality, "samples": len(rows), "inference_seconds": runtime,
                    "load_and_inference_seconds": time.monotonic() - tick, "canonical_state_sha256": state_hash,
                    "probabilities_sha256": array_sha256(p)})
                del model, p
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            full_model_rows = 2 * len(users) + 12 * len(sm["test"])
            estimated = sum(runtimes) / (2 * len(rows)) * full_model_rows
            receipt = {"stage": "skateformer_cost_smoke", "device": device, "model_rows": 2 * len(rows),
                "full_model_rows": full_model_rows, "estimated_full_inference_seconds": estimated,
                "practical_under_limit": estimated < maximum_full_seconds, "cpu_threads": threads,
                "batch": batch, "gpu_used": device == "cuda", "fine_tuning_executed": False,
                "models": model_receipts, "inputs": inputs, "source_files": source_files,
                "seconds": time.monotonic() - started}
            smoke_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt), flush=True)
            return None, receipt
        if smoke_path.exists():
            smoke = json.loads(smoke_path.read_text())
            if smoke["estimated_full_inference_seconds"] > maximum_full_seconds:
                raise RuntimeError("Selected-device smoke exceeds the declared full-run time limit")
        train_probability = np.zeros((len(users), 40), np.float64)
        test_views = []
        test_rows = np.arange(len(sm["test"]), dtype=np.int64)
        for fold, rows in enumerate(fold_indices):
            held_views = []
            for modality in ("joint", "bone"):
                tick = time.monotonic()
                model, state_hash = load_model(spec["folds"][fold][modality], device=device)
                held_views.append(predict(model, data["train"], counts["train"], rows, modality, batch=batch))
                ptest = predict(model, data["test"], counts["test"], test_rows, modality, batch=batch)
                test_views.append(normalize(ptest))
                info = {"fold": fold, "modality": modality, "held_rows": len(rows),
                        "test_rows": len(test_rows), "seconds": time.monotonic() - tick,
                        "canonical_state_sha256": state_hash}
                model_receipts.append(info)
                print(json.dumps(info), flush=True)
                del model, ptest
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            # Preserve the original float32 modality mean before float64 normalization.
            train_probability[rows] = normalize(((held_views[0] + held_views[1]) * .5).astype(np.float32))
        test_probability = normalize(np.mean(np.stack(test_views, axis=0), axis=0))
    arrays = {"train_skate": train_probability[orders["train"]], "test_skate": test_probability[orders["test"]],
        "labels": labels[orders["train"]], "users": users[orders["train"]],
        "train_groups": cm["train"].group.to_numpy(dtype=str), "test_groups": cm["test"].group.to_numpy(dtype=str)}
    np.savez_compressed(target, **arrays)
    report = {"stage": "skateformer", "array_file": target.name, "array_sha256": sha256(target),
        "array_bytes": target.stat().st_size, "inputs": inputs, "source_files": source_files,
        "learned_inputs_used": checkpoints, "method": {"held_users": FOLDS,
            "train": "Each held fold uses its own joint/bone FP16 replay states, float32 mean then float64 normalization",
            "test": "Equal mean of all twelve normalized float64 model probabilities, followed by normalization",
            "fold0": "Semantic experiment control and backbone experiment treatment state equivalence is verified separately",
            "preprocessing": "Fixed H36M17-to-NTU25 mapping, translated and duplicated actor, 0.95 center crop, 64-frame interpolation, NTU24 partition",
            "fine_tuning_executed": False, "probability_cache_input": False},
        "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": array_sha256(a)} for k, a in arrays.items()},
        "models": model_receipts, "seconds": time.monotonic() - started,
        "runtime": {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "torchvision", "timm", "safetensors", "threadpoolctl")},
            "cpu_threads": threads, "batch": batch, "device": device, "gpu_used": device == "cuda",
            "cuda_version": torch.version.cuda if device == "cuda" else None,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32), "autocast_enabled": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None},
        "technical_scope": "Saved-state inference with the configured preprocessing and runtime."}
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "skateformer", "output": str(target), "seconds": report["seconds"]}), flush=True)
    return target, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("skeleton-cache", "skeleton-train-meta", "skeleton-test-meta", "train-meta", "test-meta", "checkpoint-map", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument("--maximum-full-seconds", type=float, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run_stage(**vars(parser.parse_args()))
