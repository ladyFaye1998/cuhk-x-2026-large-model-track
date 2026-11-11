"""Original five-fold raw-adaptive training on explicit CUDA with retained FP32 states."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
import pandas as pd
import torch

import raw_adaptive_vendor as core
import raw_adaptive_cuda_backend as backend


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def input_files(input_root):
    root = Path(input_root).resolve(strict=True)
    result = {}
    for key, name in (("train_index", "nonvisual_entries.jsonl.gz"),
                      ("test_index", "test_nonvisual_entries.jsonl.gz"),
                      ("template", "sample_submission.csv")):
        matches = sorted(root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name} beneath --input-root; found {len(matches)}")
        result[key] = matches[0]
    return result


def prepare_raw(files, output):
    """Execute the original archive stores, body tracking and resampling on CPU."""
    core.OUTPUT_ROOT = Path(output)
    train_entries = [row for row in core.read_index(files["train_index"]) if core.is_skeleton(row)]
    test_entries = [row for row in core.read_index(files["test_index"]) if core.is_skeleton(row)]
    if not train_entries or not test_entries:
        raise ValueError("Official archive indexes contain no Skeleton observations")
    if not {row["disk"] for row in train_entries}.issubset(core.TRAIN_DISK_IDS):
        raise ValueError("Training index contains an unexpected archive disk")
    print(json.dumps({"stage": "raw_adaptive_preparation", "train_json_entries": len(train_entries),
                      "test_json_entries": len(test_entries)}), flush=True)
    stores, test_store, range_receipts = {}, None, []
    original_fetch = core.fetch_range

    def recorded_fetch(file_id, start, end, attempts=8):
        payload = original_fetch(file_id, start, end, attempts)
        range_receipts.append({"file_id": file_id, "start": start, "end": end,
                               "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        return payload

    core.fetch_range = recorded_fetch
    try:
        for disk in sorted({row["disk"] for row in train_entries}):
            entries = [row for row in train_entries if row["disk"] == disk]
            stores[disk] = core.ContiguousStore(core.TRAIN_DISK_IDS[disk], entries, f"adaptive_skeleton_train_{disk}")

        class TrainStore:
            def raw(self, entry):
                return stores[entry["disk"]].raw(entry)

        train_groups, train_raw, train_valid = core.build_split(TrainStore(), train_entries, "train")
        test_store = core.SparseStore(core.TEST_ID, test_entries)
        test_groups, test_raw, test_valid = core.build_split(test_store, test_entries, "test")
    finally:
        core.fetch_range = original_fetch
        for store in stores.values():
            store.close()
        if test_store is not None:
            test_store.close()
    sample = pd.read_csv(files["template"], usecols=["path"])
    paths = sample.path.astype(str).tolist()
    labels = np.asarray([int(group.split("/", 1)[0].split("_", 1)[0]) for group in train_groups], dtype=np.int64)
    users = np.asarray([int(re.search(r"user(\d+)", group).group(1)) for group in train_groups], dtype=np.int64)
    if len(labels) != 2931 or len(test_groups) != 405 or len(paths) != 405:
        raise ValueError("Unexpected official train/test row count")
    if [path.strip("/").rsplit("/", 1)[-1] for path in paths] != test_groups or len(set(paths)) != 405:
        raise ValueError("Official test template order does not match decoded observations")
    if not train_valid.all() or not test_valid.all() or set(labels) != set(range(core.N_CLASSES)):
        raise ValueError("Incomplete Skeleton observations or class coverage")
    if set(users) != set(sum((list(held) for held in core.FOLDS), [])):
        raise ValueError("Training users differ from the original five-fold protocol")
    receipt = {"train_rows": len(labels), "test_rows": len(test_groups),
               "raw_shape": [core.N_FRAMES, core.N_JOINTS, core.N_CHANNELS],
               "range_receipts": sorted(range_receipts, key=lambda row: (row["file_id"], row["start"], row["end"])),
               "tensor_sha256": {"train": hashlib.sha256(train_raw.tobytes()).hexdigest(),
                                 "test": hashlib.sha256(test_raw.tobytes()).hexdigest()}}
    write_json(Path(output) / "raw_observation_receipt.json", receipt)
    return train_groups, test_groups, train_raw, test_raw, train_valid, test_valid, labels, users


def initialize_cuda():
    """Initialize explicit CUDA FP32 only after prepared observations are verified."""
    return backend.initialize_cuda(core)


def save_state(path, state, metadata, half=False):
    """Persist the selected host state without altering the live training model."""
    floating = {str(value.dtype) for value in state.values() if value.is_floating_point()}
    if floating != {"torch.float32"}:
        raise RuntimeError(f"Expected the original FP32 training state, received {floating}")
    saved = {key: value.half() if half and value.is_floating_point() else value for key, value in state.items()}
    payload = {"schema_version": "cuhkx-raw-skeleton-adaptive-2stream-xla/v3" if half else "cuhkx-raw-adaptive-complete-training/v1", "state_dict": saved,
               "frames": core.N_FRAMES, "pretrained_backbone": False,
               "streams": ["root-centered-joint", "bone"], "projection_head_required_at_inference": False,
               "execution_backend": "cuda", "bit_identity_to_original_tpu": "not asserted",
               **metadata}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
            "floating_dtype": "float16" if half else "float32", **metadata}


def run_raw_adaptive_stage(input_root, out_dir, prepared_root=None):
    """Train every original fold/full fit on CUDA from verified prepared observations."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required for complete training; no CPU or saved-state fallback exists")
    if prepared_root is None:
        raise ValueError("The CUDA stage requires --prepared-root; raw downloads are performed separately on CPU")
    output = Path(out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; existing artifacts are preserved")
    files = input_files(input_root)
    output.mkdir(parents=True, exist_ok=True)
    core.INPUT_ROOT, core.OUTPUT_ROOT = Path(input_root).resolve(), output
    started = time.time()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    inputs = [{"path": str(path.resolve()), "sha256": sha256(path), "role": role}
              for role, path in files.items()]
    source_files = [{"path": str(Path(path).resolve()), "sha256": sha256(path)}
                    for path in (__file__, core.__file__, backend.__file__)]
    protocol = {"stage": "raw_adaptive", "inputs": inputs, "source_files": source_files,
                "learned_inputs_used": [], "method": "Original five-fold supervised raw Skeleton training with explicit CUDA backend and median-selected full fit",
                "execution_backend": "cuda", "mixed_precision": False, "tf32": False,
                "backend_difference": "CUDA eager optimizer/device/save synchronization and CUDA device RNG replace XLA operations; model, loss, optimizer formulas and numeric seeds are retained",
                "bit_identity_to_original_tpu": "not asserted: CUDA RNG and floating-point kernels differ",
                "original_source_sha256": core.ORIGINAL_SOURCE_SHA256,
                "folds": core.FOLDS, "fold_seeds": [20260931 + fold for fold in range(len(core.FOLDS))],
                "fullfit_seed": 20261031, "maximum_fold_epochs": core.EPOCHS,
                "fullfit_epoch_rule": "round(median(selected_fold_epochs))",
                "validation_from_epoch": core.VALIDATE_FROM, "validation_every": core.VALIDATE_EVERY,
                "patience_checks": core.PATIENCE, "batch_size": core.BATCH_SIZE,
                "evaluation_batch_size": core.EVAL_BATCH_SIZE,
                "math_changes": [], "operational_changes": ["Explicit CUDA FP32 eager backend in place of XLA", "Verified prepared observations required; no accelerator-time downloads", "Retain each selected OOF FP32 state and full-fit FP32 state", "Canonical output schema and receipts"],
                "validation_limit": "Historical folds select checkpoints; this is the original model-selection protocol, not an independent untouched evaluation."}
    write_json(output / "stage_protocol.json", protocol)
    if prepared_root is None:
        observations = prepare_raw(files, output)
    else:
        from raw_adaptive_prepare import load_prepared
        observations, prepared_manifest, receipt = load_prepared(prepared_root, files)
        write_json(output / "raw_observation_receipt.json", receipt)
        prepared_path = Path(prepared_root).resolve()
        protocol["prepared_observations"] = {
            "manifest_path": str(prepared_path / "preparation_manifest.json"),
            "manifest_sha256": sha256(prepared_path / "preparation_manifest.json"),
            "array_sha256": prepared_manifest["array_sha256"],
            "decoder_source_sha256": prepared_manifest["decoder_source_sha256"],
            "learned_inputs_used": [],
        }
        protocol["source_files"].append({"path": str(Path(__import__("raw_adaptive_prepare").__file__).resolve()),
                                          "sha256": sha256(__import__("raw_adaptive_prepare").__file__)})
        protocol["operational_changes"].append("Verified CPU observation preparation reused without network downloads on the accelerator")
        write_json(output / "stage_protocol.json", protocol)
    train_groups, test_groups, train_raw, test_raw, train_valid, test_valid, labels, users = observations
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "pandas": pd.__version__, **initialize_cuda()}
    write_json(output / "runtime.json", runtime)
    train_tensor, test_tensor = torch.from_numpy(train_raw), torch.from_numpy(test_raw)
    train_valid_tensor, test_valid_tensor = torch.from_numpy(train_valid[:, None]), torch.from_numpy(test_valid[:, None])
    oof_logits = np.zeros((len(labels), core.N_CLASSES), dtype=np.float32)
    folds, seen = np.full(len(labels), -1, dtype=np.int64), np.zeros(len(labels), np.int8)
    reports, best_epochs, checkpoint_receipts = [], [], []
    all_rows = np.arange(len(labels), dtype=np.int64)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for fold, heldout_users in enumerate(core.FOLDS):
        valid_rows = np.flatnonzero(np.isin(users, heldout_users))
        train_rows = np.flatnonzero(~np.isin(users, heldout_users))
        folds[valid_rows], seen[valid_rows] = fold, seen[valid_rows] + 1
        model, state, best_epoch, best_score = core.train_model(train_tensor, train_valid_tensor,
            labels, users, train_rows, valid_rows, 20260931 + fold, core.EPOCHS)
        logits = core.predict_logits(model, train_tensor, train_valid_tensor, valid_rows)
        oof_logits[valid_rows] = logits
        record = {"fold": fold, "heldout_users": list(heldout_users), "valid_rows": int(len(valid_rows)),
                  "best_epoch": int(best_epoch), "selection_accuracy": float(best_score),
                  "accuracy": float(np.mean(logits.argmax(1) == labels[valid_rows]))}
        checkpoint_receipts.append(save_state(checkpoint_dir / f"fold_{fold}_fp32.pt", state,
            {"role": "selected_oof", "fold": fold, "held_users": list(heldout_users),
             "fit_users": sorted(map(int, set(users[train_rows]))), "selected_epoch": int(best_epoch), "seed": 20260931 + fold}))
        del model, state
        reports.append(record)
        best_epochs.append(best_epoch)
        write_json(output / "fold_progress.json", {"folds": reports, "checkpoints": checkpoint_receipts})
        print(json.dumps(record), flush=True)
    if not np.all(seen == 1) or not np.isfinite(oof_logits).all():
        raise RuntimeError("Original OOF generation failed to cover every row exactly once")
    full_epochs = int(round(float(np.median(best_epochs))))
    model, state, _, _ = core.train_model(train_tensor, train_valid_tensor, labels, users,
        all_rows, None, 20261031, full_epochs)
    test_logits = core.predict_logits(model, test_tensor, test_valid_tensor,
                                     np.arange(len(test_groups), dtype=np.int64))
    if not np.isfinite(test_logits).all():
        raise RuntimeError("Nonfinite full-fit test logits")
    metadata = {"role": "fullfit", "selected_epoch": full_epochs, "seed": 20261031}
    checkpoint_receipts.append(save_state(checkpoint_dir / "fullfit_fp32.pt", state, metadata))
    half_receipt = save_state(checkpoint_dir / "raw_skeleton_adaptive_2stream_tpu_fullfit_fp16.pt", state, metadata, half=True)
    checkpoint_receipts.append(half_receipt)
    if half_receipt["bytes"] >= core.MAX_MODEL_BYTES:
        raise RuntimeError("Original FP16 checkpoint exceeds its size bound")
    array_path = output / "teacher_arrays.npz"
    temporary = output / "teacher_arrays.tmp.npz"
    np.savez_compressed(temporary, train_raw_logits=oof_logits, test_raw_logits=test_logits,
                        labels=labels, users=users, folds=folds,
                        train_groups=np.asarray(train_groups, dtype=str), test_groups=np.asarray(test_groups, dtype=str))
    os.replace(temporary, array_path)
    manifest = {**protocol, "status": "completed", "array_file": array_path.name, "array_sha256": sha256(array_path),
                "array_shapes": {"train_raw_logits": list(oof_logits.shape), "test_raw_logits": list(test_logits.shape)},
                "runtime": runtime, "fold_results": reports, "fullfit_epochs": full_epochs,
                "backend_optimizer_steps": core.xm.optimizer_steps,
                "checkpoints": checkpoint_receipts, "elapsed_seconds": time.time() - started,
                "raw_observation_receipt_sha256": sha256(output / "raw_observation_receipt.json")}
    write_json(output / "stage_manifest.json", manifest)
    print(json.dumps({"stage": "raw_adaptive", "status": "completed", "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return array_path, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Directory containing the two official archive indexes and sample_submission.csv")
    parser.add_argument("--out", required=True, help="Empty output directory for this complete supervised run")
    parser.add_argument("--prepared-root", help="Optional completed CPU observation-preparation directory; hashes/row order/decoder must match")
    args = parser.parse_args()
    run_raw_adaptive_stage(args.input_root, args.out, args.prepared_root)
