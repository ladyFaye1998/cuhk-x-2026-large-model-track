"""Reproduce the fixed six-fold EfficientGCN transfer fitting from observations.

Full training uses unchanged functions from the recovered executors. It selects
the final epoch only, writes full FP32 states, and retains the original FP16
serialization for test replay. No learned predictions or target CSV are inputs.
Dependencies: Python 3.12, numpy 2.0.2, torch 2.10.0, threadpoolctl.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import efficientgcn_original_ops as ops
from efficientgcn_vendor import model as vendor_model

ROOT = Path(__file__).resolve().parent
ARCHITECTURE = "EfficientGCN-B4-SG-H36M17-40"
PUBLIC_SHA = "263a37af21ef80abcce4a5ac189a950473d54b379f0a820bae1c6a498aa337e6"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fact(path, role=None):
    path = Path(path).resolve()
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
    if role:
        result["role"] = role
    return result


def array_fact(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def create_target():
    # The original create_target call is unchanged apart from its package import.
    return vendor_model.create(
        "EfficientGCN-B4", data_shape=(3, 6, ops.FRAMES, 17, 1), num_class=40,
        A=torch.from_numpy(ops.h36m17_adjacency()), parts=ops.H36M17_PARTS,
        stem_channel=64, block_args=[[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]],
        fusion_stage=2, act_type="swish", att_type="stja", layer_type="SG",
        drop_prob=0.25, kernel_size=[5, 2], scale_args=[1.2, 1.35], expand_ratio=0,
        reduct_ratio=2, bias=True, edge=True)


def read_metadata(path, *, training):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not r.get("group") for r in rows):
        raise ValueError("Missing metadata groups")
    groups = np.asarray([r["group"].replace("\\", "/") for r in rows], dtype=str)
    if len(set(groups)) != len(groups):
        raise ValueError("Duplicate metadata groups")
    values = {"groups": groups}
    if training:
        values["labels"] = np.asarray([int(r["label"]) for r in rows], np.int64)
        values["users"] = np.asarray([int(r["user"].removeprefix("user")) for r in rows], np.int64)
        if set(values["labels"]) != set(range(40)):
            raise ValueError("Expected forty official training classes")
    if "n_valid" in rows[0]:
        values["n_valid"] = np.asarray([int(r["n_valid"]) for r in rows], np.int32)
    return values


def load_training_observations(cache, metadata):
    meta = read_metadata(metadata, training=True)
    with np.load(cache, allow_pickle=False) as archive:
        if set(archive.files) not in ({"train", "train_n_valid"},
                                     {"train", "train_n_valid", "test", "test_n_valid"}):
            raise ValueError("Only raw skeleton observation arrays are allowed")
        skeleton, valid = archive["train"], archive["train_n_valid"]
    if skeleton.shape != (2931, 128, 17, 3) or skeleton.dtype != np.float32 or valid.shape != (2931,):
        raise ValueError("Unexpected official training observation shape or dtype")
    if not np.isfinite(skeleton).all() or np.any((valid < 1) | (valid > 128)):
        raise ValueError("Invalid skeleton observations")
    if len(meta["groups"]) != 2931 or not np.array_equal(valid, meta["n_valid"]):
        raise ValueError("Skeleton metadata alignment mismatch")
    return skeleton, valid.astype(np.int32), meta


def verify_folds(meta):
    contract = json.loads((ROOT / "fold_contract.json").read_text())
    result = []
    for record in contract["canonical_folds"]:
        held = np.flatnonzero(np.isin(meta["users"], record["users"]))
        fitting = np.flatnonzero(~np.isin(meta["users"], record["users"]))
        group_bytes = ("\n".join(meta["groups"][held]) + "\n").encode()
        if (len(held) != record["expected_rows"]
                or hashlib.sha256(held.astype("<i8").tobytes()).hexdigest() != record["held_indices_int64_le_sha256"]
                or hashlib.sha256(fitting.astype("<i8").tobytes()).hexdigest() != record["training_indices_int64_le_sha256"]
                or hashlib.sha256(group_bytes).hexdigest() != record["held_group_sequence_utf8_lf_sha256"]):
            raise ValueError("Recorded fold membership or canonical skeleton order changed")
        result.append((record, fitting, held))
    if not np.array_equal(np.sort(np.concatenate([x[2] for x in result])), np.arange(2931)):
        raise ValueError("Incomplete held-fold coverage")
    return result


def source_facts():
    files = [Path(__file__).resolve(), ROOT / "efficientgcn_original_ops.py", ROOT / "fold_contract.json"]
    files += sorted((ROOT / "efficientgcn_vendor").rglob("*.py"))
    return [fact(p) for p in files]


def runtime(device):
    return {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p)
        for p in ("numpy", "torch", "threadpoolctl")}, "device": str(device),
        "cuda_version": torch.version.cuda, "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "autocast_enabled": False, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "original_runtime_match": np.__version__ == "2.0.2" and torch.__version__ == "2.10.0+cu128"}


def configure(device, threads):
    if device not in ("cpu", "cuda") or threads not in (1, 2):
        raise ValueError("Select cpu/cuda and one or two CPU threads")
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA request cannot be satisfied")
    torch.set_num_threads(threads)
    ops.seed_everything(ops.SEED)
    return torch.device(device)


class ProgressDataset(ops.SkeletonDataset):
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if self.training:
            print(json.dumps({"event": "training_epoch", "fold": self.fold, "arm": self.arm,
                              "epoch": epoch + 1, "epochs": ops.EPOCHS}), flush=True)


def canonical_order(source, target_path, *, training):
    if target_path is None:
        return np.arange(len(source["groups"])), source
    target = read_metadata(target_path, training=training)
    lookup = {g: i for i, g in enumerate(source["groups"])}
    if set(lookup) != set(target["groups"]):
        raise ValueError("Canonical row identifiers do not match skeleton observations")
    order = np.asarray([lookup[g] for g in target["groups"]], np.int64)
    if training and any(not np.array_equal(source[k][order], target[k]) for k in ("labels", "users")):
        raise ValueError("Canonical training labels or subjects differ")
    return order, target


def predict_test(model, skeleton, valid):
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(skeleton), ops.BATCH_SIZE):
            features = np.stack([ops.multi_input(ops.normalize_and_augment(
                ops.temporal_sample(skeleton[i], int(valid[i]), False, ops.SEED + i * 101),
                False, ops.SEED + i * 101)) for i in range(start, min(start + ops.BATCH_SIZE, len(skeleton)))])
            logits, _ = model(torch.from_numpy(features).to(device))
            predictions.append(torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float64))
    result = np.concatenate(predictions)
    return result / result.sum(1, keepdims=True)


def cpu_smoke(skeleton, valid, meta, public_state, output, inputs, *, samples=8):
    """Two deterministic one-minibatch fits; never emits a complete fold artifact."""
    if not 2 <= samples <= 8:
        raise ValueError("CPU smoke is bounded to eight fitting and held examples")
    _, fitting, held = verify_folds(meta)[0]
    results = []
    old_epochs, old_loader = ops.EPOCHS, ops.make_loader

    def smoke_loader(dataset, epoch, shuffle):
        generator = torch.Generator().manual_seed(ops.SEED + epoch * 65_537)
        return torch.utils.data.DataLoader(dataset, batch_size=ops.BATCH_SIZE, shuffle=shuffle,
            drop_last=False, num_workers=0, pin_memory=False, generator=generator, persistent_workers=False)

    try:
        ops.EPOCHS, ops.make_loader = 1, smoke_loader
        for repeat in range(2):
            ops.seed_everything(ops.SEED)
            model = create_target()
            partial = ops.apply_transfer(model, public_state)
            initial = state_sha(model.state_dict())
            ds = ops.SkeletonDataset(skeleton, valid, meta["labels"], fitting[:samples], True)
            hs = ops.SkeletonDataset(skeleton, valid, meta["labels"], held[:samples], False)
            ops.seed_everything(ops.SEED)
            started = time.monotonic()
            summary, probabilities, state = ops.train_arm("bounded_smoke", model, ds, hs, torch.device("cpu"))
            elapsed = time.monotonic() - started
            assert summary["epochs"] == 1 and len(summary["history"]) == 1
            assert summary["history"][0]["learning_rate"] == .015
            assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
            assert all(v.dtype == torch.float32 for v in state.values() if v.is_floating_point())
            final = state_sha(state)
            assert initial != final and probabilities.shape == (samples, 40) and np.isfinite(probabilities).all()
            buffer = io.BytesIO()
            torch.save({"model": state, "architecture": ARCHITECTURE}, buffer)
            restored = torch.load(io.BytesIO(buffer.getvalue()), map_location="cpu", weights_only=True)["model"]
            assert state_sha(restored) == final
            results.append({"repeat": repeat, "fitting_samples": samples, "held_samples": samples,
                "optimizer_steps": 1, "seconds": elapsed, "initial_state_sha256": initial,
                "final_state_sha256": final, "probabilities": array_fact(probabilities),
                "fp32_checkpoint_bytes": len(buffer.getvalue()), "fp32_roundtrip_exact": True})
            del model, state, restored, buffer
            gc.collect()
        assert results[0]["final_state_sha256"] == results[1]["final_state_sha256"]
        assert results[0]["probabilities"] == results[1]["probabilities"]
        report = {"stage": "efficientgcn_training_smoke", "complete_training": False, "gpu_used": False,
            "partial_load": {k: v for k, v in partial.items() if k != "records"}, "runs": results,
            "repeated_fit_byte_identical": True, "inputs": inputs, "source_files": source_facts(),
            "runtime": runtime(torch.device("cpu")), "smoke_resource_changes": {"epochs": 1, "workers": 0, "fitting_samples": samples},
            "full_recipe_unchanged": {"epochs": old_epochs, "batch_size": ops.BATCH_SIZE, "workers": 2},
            "full_gpu_training_started": False}
        write_json(output / "cpu_training_smoke.json", report)
        print(json.dumps({"stage": report["stage"], "success": True, "report": str(output / "cpu_training_smoke.json"),
                          "seconds": sum(r["seconds"] for r in results)}), flush=True)
        return report
    finally:
        ops.EPOCHS, ops.make_loader = old_epochs, old_loader


def train(skeleton_cache, skeleton_train_meta, public_checkpoint, output_dir, *,
          canonical_train_meta=None, skeleton_test_meta=None, canonical_test_meta=None,
          device="cpu", test_device="cpu", threads=2, arms="transfer", smoke=False):
    started = time.monotonic()
    if arms not in ("transfer", "both"):
        raise ValueError("Choose transfer or both fixed arms")
    selected_device = configure(device, threads)
    if test_device not in ("cpu", "cuda"):
        raise ValueError("Choose cpu/cuda for the separate test replay")
    if test_device == "cuda" and device != "cuda":
        raise ValueError("CUDA test replay requires an explicitly configured CUDA run")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory to preserve previous artifacts")
    output.mkdir(parents=True, exist_ok=True)
    public_checkpoint = Path(public_checkpoint).resolve()
    if public_checkpoint.stat().st_size != 9608526 or sha(public_checkpoint) != PUBLIC_SHA:
        raise ValueError("Public initialization fingerprint mismatch")
    inputs = [fact(skeleton_cache, "official_skeleton_observations"),
              fact(skeleton_train_meta, "official_training_metadata"), fact(public_checkpoint, "public_initialization")]
    for path, role in ((canonical_train_meta, "canonical_training_metadata"),
                       (skeleton_test_meta, "official_test_row_metadata"), (canonical_test_meta, "canonical_test_row_metadata")):
        if path is not None:
            inputs.append(fact(path, role))
    skeleton, valid, meta = load_training_observations(skeleton_cache, skeleton_train_meta)
    folds = verify_folds(meta)
    public_state = ops.safe_load_official_checkpoint(public_checkpoint)
    with threadpool_limits(limits=threads):
        if smoke:
            if device != "cpu":
                raise ValueError("This bounded training smoke is CPU-only")
            return cpu_smoke(skeleton, valid, meta, public_state, output, inputs)
        train_oof = np.zeros((len(skeleton), 40), np.float32)
        control_oof = np.zeros_like(train_oof) if arms == "both" else None
        trained = []
        # Each fold recreates the same common target initialization and resets
        # the seed before fitting, exactly as its independent original executor.
        for record, fitting, held in folds:
            fold = record["fold"]
            fold_dir = output / f"fold{fold}"
            fold_dir.mkdir()
            print(json.dumps({"event": "fold_started", "fold": fold, "fitting_rows": len(fitting), "held_rows": len(held)}), flush=True)
            ops.seed_everything(ops.SEED)
            base = create_target()
            base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
            initial_state_sha = state_sha(base_state)
            del base
            summary_by_arm = {}
            for arm in (("control", "transfer") if arms == "both" else ("transfer",)):
                model = create_target()
                model.load_state_dict(base_state, strict=True)
                partial = ops.apply_transfer(model, public_state) if arm == "transfer" else None
                ds = ProgressDataset(skeleton, valid, meta["labels"], fitting, True)
                ds.fold, ds.arm = fold, arm
                hs = ops.SkeletonDataset(skeleton, valid, meta["labels"], held, False)
                ops.seed_everything(ops.SEED)
                tick = time.monotonic()
                summary, probabilities, state = ops.train_arm(arm, model, ds, hs, selected_device)
                summary["seconds"] = time.monotonic() - tick
                summary_by_arm[arm] = summary
                if arm == "control":
                    control_oof[held] = probabilities
                else:
                    train_oof[held] = probabilities
                    if any(v.dtype != torch.float32 for v in state.values() if v.is_floating_point()):
                        raise ValueError("Full FP32 fold state was not preserved")
                    fp32_path = fold_dir / f"efficientgcn_fold{fold}_fp32.pt"
                    torch.save({"model": state, "architecture": ARCHITECTURE}, fp32_path)
                    fp16_path = fold_dir / f"efficientgcn_fold{fold}_fp16.pt"
                    fp16_path.write_bytes(ops.serialize_half_checkpoint(state))
                    write_json(fold_dir / "partial_load_manifest.json", partial)
                    trained.append({"fold": fold, "held_users": record["users"], "initial_state_sha256": initial_state_sha,
                        "fp32": fact(fp32_path), "fp32_canonical_state_sha256": state_sha(state), "fp16": fact(fp16_path),
                        "held_probabilities": array_fact(probabilities), "summary": summary})
                del model, state
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            write_json(fold_dir / "training_report.json", {"fold": record, "arms": summary_by_arm})
        order, canonical = canonical_order(meta, canonical_train_meta, training=True)
        normalized = train_oof.astype(np.float64)
        normalized = np.clip(normalized, 1e-12, np.inf)
        normalized /= normalized.sum(1, keepdims=True)
        arrays = {"train_egcn": normalized[order], "train_transfer_fp32_oof": train_oof[order],
                  "labels": canonical["labels"], "users": canonical["users"], "train_groups": canonical["groups"]}
        if control_oof is not None:
            arrays["train_control_fp32_oof"] = control_oof[order]
        if skeleton_test_meta is not None:
            tm = read_metadata(skeleton_test_meta, training=False)
            with np.load(skeleton_cache, allow_pickle=False) as archive:
                test, test_valid = archive["test"], archive["test_n_valid"]
            if (test.shape != (405, 128, 17, 3) or test.dtype != np.float32
                    or not np.isfinite(test).all() or not np.array_equal(test_valid, tm["n_valid"])):
                raise ValueError("Invalid official test observation alignment")
            test_views = []
            for record in trained:
                model = create_target()
                raw = torch.load(record["fp16"]["path"], map_location="cpu", weights_only=True)
                model.load_state_dict(raw["model"], strict=True)
                model.to(torch.device(test_device)).eval()
                test_views.append(predict_test(model, test, test_valid))
                del model, raw
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            mean = np.mean(np.stack(test_views), axis=0)
            mean /= mean.sum(1, keepdims=True)
            test_order, test_canonical = canonical_order(tm, canonical_test_meta, training=False)
            arrays["test_egcn"] = mean[test_order]
            arrays["test_groups"] = test_canonical["groups"]
        target = output / "teacher_arrays.npz"
        target.write_bytes(ops.deterministic_npz_bytes(arrays))
    report = {"stage": "efficientgcn", "operation": "sixfold_transfer_training",
        "status": "all_six_folds_fully_trained", "array_file": target.name, "array_sha256": sha(target),
        "array_bytes": target.stat().st_size, "inputs": inputs, "source_files": source_facts(),
        "learned_inputs_used": [fact(public_checkpoint, "public_initialization")], "folds": trained,
        "arrays": {k: array_fact(v) for k, v in arrays.items()}, "runtime": runtime(selected_device),
        "seconds": time.monotonic() - started, "method": {"seed": ops.SEED, "epochs": ops.EPOCHS,
            "batch_size": ops.BATCH_SIZE, "optimizer": "SGD Nesterov momentum 0.9 weight_decay 0.0001",
            "base_learning_rate": ops.BASE_LR, "schedule": "2-epoch warmup then cosine to zero",
            "selection": "Final epoch only; no held-driven early stopping or parameter search",
            "arms": arms, "training_executed": True, "fine_tuning_executed": True, "prediction_cache_input": False,
            "test_replay": "Six FP16 serialized final states promoted to FP32, normalized model probabilities averaged equally; no probability floor on test",
            "test_device": test_device, "original_test_device": "cpu",
            "original_hypothesis_gates_re_evaluated": False},
        "runtime_note": "The fixed fitting recipe records its environment and generated model state."}
    write_json(output / "stage_manifest.json", report)
    print(json.dumps({"stage": report["stage"], "output": str(target), "seconds": report["seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("skeleton-cache", "skeleton-train-meta", "public-checkpoint", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("canonical-train-meta", "skeleton-test-meta", "canonical-test-meta"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--test-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", choices=(1, 2), type=int, default=2)
    parser.add_argument("--arms", choices=("transfer", "both"), default="transfer")
    parser.add_argument("--smoke", action="store_true")
    train(**vars(parser.parse_args()))
