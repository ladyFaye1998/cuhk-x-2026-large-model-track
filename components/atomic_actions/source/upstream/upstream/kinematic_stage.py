"""Train or replay the kinematic teacher from explicit official observations.

Fit mode runs the complete recovered six-fold and full-fit training sources.
Replay mode regenerates probabilities from explicit model checkpoints and does
not claim those checkpoints were fitted in the current run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import kinematic_fullfit as model_source
from assemble_teacher_inputs import FOLDS, read_metadata, sha


HERE = Path(__file__).resolve().parent


def receipt(path, role):
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "role": role}


def run_kinematic_stage(features, train_meta, test_meta, out, mode="fit", fold_dir=None,
                        fullfit_checkpoint=None, threads=2):
    started = time.perf_counter()
    features, out = Path(features), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "teacher_arrays.npz").exists():
        raise FileExistsError("Use a new output directory for this stage")
    if mode not in {"fit", "replay"}:
        raise ValueError(mode)
    train, test = read_metadata(train_meta), read_metadata(test_meta)
    with np.load(features, allow_pickle=False) as data:
        labels = data["train_labels"].astype(np.int64)
        users = data["train_users"].astype(np.int64)
        if not np.array_equal(labels, [int(r["label"]) for r in train]):
            raise ValueError("Observation labels and official metadata disagree")
        if not np.array_equal(users, [int(r["user"]) for r in train]):
            raise ValueError("Observation users and official metadata disagree")
        if not np.array_equal(data["train_trials"], [r["trial"] for r in train]):
            raise ValueError("Observation trials and official metadata disagree")
        if not np.array_equal(data["test_numbers"], [int(r["test_number"]) for r in test]):
            raise ValueError("Test observation order and official metadata disagree")
        raw_train = model_source.kinematic_features(data["train_segments"].astype(np.float32))
        raw_test = model_source.kinematic_features(data["test_segments"].astype(np.float32))
        available_train = torch.from_numpy(data["train_masks"][:, 2:3].astype(np.float32))
        available_test = torch.from_numpy(data["test_masks"][:, 2:3].astype(np.float32))
    inputs = [receipt(features, "official_observation_features"),
              receipt(train_meta, "official_train_metadata"), receipt(test_meta, "official_test_metadata")]
    if mode == "fit":
        if fold_dir is not None or fullfit_checkpoint is not None:
            raise ValueError("Fit mode does not take existing learned checkpoints")
        fold_dir = out / "models" / "cv"
        full_dir = out / "models" / "fullfit"
        for source, destination in (("kinematic_cv.py", fold_dir), ("kinematic_fullfit.py", full_dir)):
            subprocess.run([sys.executable, str(HERE / source), "--features", str(features.resolve()),
                            "--out", str(destination.resolve())], check=True)
        fullfit_checkpoint = full_dir / "skeleton_kinematic_fullfit.pt"
    elif fold_dir is None or fullfit_checkpoint is None:
        raise ValueError("Replay mode requires --fold-dir and --fullfit-checkpoint")
    fold_dir, fullfit_checkpoint = Path(fold_dir), Path(fullfit_checkpoint)
    torch.set_num_threads(threads)
    oof = np.zeros((len(labels), 40), np.float32)
    seen = np.zeros(len(labels), np.int8)
    checkpoints, fold_reports = [], []
    for fold, expected_held in enumerate(FOLDS):
        path = fold_dir / f"fold_{fold}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if list(state["held_users"]) != list(expected_held):
            raise ValueError(f"Fold {fold} has unexpected held subjects")
        held = np.isin(users, expected_held)
        fit = ~held
        if sorted(set(map(int, users[fit]))) != list(state["fit_users"]):
            raise ValueError(f"Fold {fold} has unexpected fitting subjects")
        mean, std = state["normalization_mean"].numpy(), state["normalization_std"].numpy()
        if not np.array_equal(mean, raw_train[fit].mean((0, 1), keepdims=True)):
            raise ValueError(f"Fold {fold} normalization mean differs from its fitting observations")
        if not np.array_equal(std, np.maximum(raw_train[fit].std((0, 1), keepdims=True), 1e-4)):
            raise ValueError(f"Fold {fold} normalization std differs from its fitting observations")
        values = torch.from_numpy(np.clip((raw_train[held] - mean) / std, -7, 7).astype(np.float32))
        model = model_source.SkeletonKinematicNet(state["channels"])
        model.load_state_dict(state["state_dict"], strict=True)
        oof[held] = model_source.probabilities(model, values, available_train[held])
        seen += held
        checkpoints.append(receipt(path, "fitted_kinematic_fold"))
        fold_reports.append({"fold": fold, "held_users": list(expected_held),
                             "fit_users": list(state["fit_users"]), "best_epoch": state["best_epoch"],
                             "held_rows": int(held.sum()),
                             "held_correct": int((oof[held].argmax(1) == labels[held]).sum())})
    if not np.all(seen == 1):
        raise ValueError("Every official training row must receive one held-fold prediction")
    state = torch.load(fullfit_checkpoint, map_location="cpu", weights_only=True)
    mean, std = state["normalization_mean"].numpy(), state["normalization_std"].numpy()
    if not np.array_equal(mean, raw_train.mean((0, 1), keepdims=True)):
        raise ValueError("Full-fit normalization mean differs from training observations")
    if not np.array_equal(std, np.maximum(raw_train.std((0, 1), keepdims=True), 1e-4)):
        raise ValueError("Full-fit normalization std differs from training observations")
    model = model_source.SkeletonKinematicNet(state["channels"])
    model.load_state_dict(state["state_dict"], strict=True)
    values = torch.from_numpy(np.clip((raw_test - mean) / std, -7, 7).astype(np.float32))
    test_prob = model_source.probabilities(model, values, available_test)
    checkpoints.append(receipt(fullfit_checkpoint, "fitted_kinematic_fullfit"))
    array_path = out / "teacher_arrays.npz"
    np.savez_compressed(array_path, train_kinematic=oof, test_kinematic=test_prob,
                        labels=labels, users=users,
                        train_groups=np.asarray([r["group"] for r in train], dtype=str),
                        test_groups=np.asarray([r["group"] for r in test], dtype=str))
    learned_inputs = checkpoints if mode == "replay" else []
    manifest = {"stage": "kinematic", "array_file": array_path.name, "array_sha256": sha(array_path),
                "inputs": inputs + learned_inputs,
                "source_files": [receipt(HERE / name, "source") for name in
                                 ("kinematic_stage.py", "kinematic_cv.py", "kinematic_fullfit.py", "assemble_teacher_inputs.py")],
                "learned_inputs_used": learned_inputs,
                "generated_checkpoints": checkpoints if mode == "fit" else [],
                "method": "six disjoint subject folds for training rows; separate all-training-subject full fit for test rows",
                "mode": mode, "training_executed_in_this_run": mode == "fit",
                "folds": fold_reports, "fullfit_epochs": int(state["epochs"]),
                "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__},
                "seconds": time.perf_counter() - started,
                "historical_probability_identity": "not_assessed", "competition_compliance": "not_assessed"}
    (out / "stage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--train-meta", required=True)
    parser.add_argument("--test-meta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("fit", "replay"), default="fit")
    parser.add_argument("--fold-dir")
    parser.add_argument("--fullfit-checkpoint")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_kinematic_stage(args.features, args.train_meta, args.test_meta, args.out,
          args.mode, args.fold_dir, args.fullfit_checkpoint, args.threads), indent=2))
