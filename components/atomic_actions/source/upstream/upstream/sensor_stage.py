"""Fresh six-fold CPU sensor fitting with the fixed recorded sequence recipe."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
import joblib
import lightgbm
import numba
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

import sensor_sequence_ops as sequence

FOLDS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (16, 17, 18), (19, 20, 21), (22, 23, 24))
PINNED = {"numpy": "2.0.2", "scipy": "1.15.3", "sklearn": "1.6.1",
          "pandas": "2.2.3", "lightgbm": "4.6.0", "numba": "0.61.2"}
RECIPE = {"lightgbm_weight": 0.575, "extra_trees_weight": 0.425,
          "gap_seconds": 900, "hold_penalty": 0.6, "skip_penalty": 1.1, "top_k": 3}


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


def runtime_versions():
    return {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "lightgbm": lightgbm.__version__, "numba": numba.__version__}


def validate_observations(feature_path, train_meta_path, test_meta_path):
    train_meta, test_meta = pd.read_csv(train_meta_path), pd.read_csv(test_meta_path)
    for metadata, name in ((train_meta, "train"), (test_meta, "test")):
        required = {"group", "start_time", "end_time"} | ({"label", "user"} if name == "train" else {"test_number"})
        if not required.issubset(metadata.columns):
            raise ValueError(f"Missing {name} metadata fields: {sorted(required - set(metadata.columns))}")
        if metadata.group.isna().any() or not metadata.group.astype(str).is_unique:
            raise ValueError(f"Invalid {name} row IDs")
    with np.load(feature_path, allow_pickle=False) as archive:
        expected = {"train_segments", "train_global", "train_masks", "train_labels", "train_users", "train_trials",
                    "test_segments", "test_global", "test_masks", "test_numbers"}
        if set(archive.files) != expected:
            raise ValueError("Expected the observation-only small_sensor_features.npz schema")
        train = np.nan_to_num(archive["train_global"].astype(np.float32))
        test = np.nan_to_num(archive["test_global"].astype(np.float32))
        labels, users = archive["train_labels"].astype(np.int64), archive["train_users"].astype(np.int64)
        test_numbers = archive["test_numbers"].astype(np.int64)
    if train.shape != (2931, 1723) or test.shape != (405, 1723):
        raise ValueError("Official observation feature shape mismatch")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Nonfinite observation features after the recorded nan_to_num conversion")
    if not np.array_equal(train_meta.label.to_numpy(np.int64), labels):
        raise ValueError("Training label/feature alignment mismatch")
    if not np.array_equal(train_meta.user.to_numpy(np.int64), users):
        raise ValueError("Training subject/feature alignment mismatch")
    if not np.array_equal(test_meta.test_number.to_numpy(np.int64), test_numbers):
        raise ValueError("Test row/feature alignment mismatch")
    if set(users) != set(sum((list(held) for held in FOLDS), [])) or set(labels) != set(range(40)):
        raise ValueError("Unexpected official classes or subject partition")
    return train, test, labels, users, train_meta, test_meta


def run_sensor_stage(feature_path, train_meta_path, test_meta_path, out_dir, save_models=False, threads=4):
    """Return (teacher_arrays.npz path, stage_manifest dict); inputs are observations only.

    The sequence recipe is fixed before this run. Its output is a separate
    teacher view; it is not multiplied by the historical display-blend weight.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    versions = runtime_versions()
    if versions != PINNED:
        raise RuntimeError(f"Install requirements-sensor.txt for the recorded runtime; observed {versions}")
    output = Path(out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; existing artifacts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    inputs = [{"path": str(Path(path).resolve()), "sha256": sha256(path), "role": role}
              for path, role in ((feature_path, "official_observation_features"),
                                 (train_meta_path, "official_training_metadata"),
                                 (test_meta_path, "official_test_metadata"))]
    source_files = [{"path": str(Path(path).resolve()), "sha256": sha256(path)}
                    for path in (__file__, sequence.__file__)]
    protocol = {"stage": "sensor", "inputs": inputs, "source_files": source_files, "learned_inputs_used": [],
                "runtime": versions, "threads": threads, "folds": FOLDS, "recipe": RECIPE,
                "hyperparameter_search_performed": False,
                "method": "Fresh per-fold LightGBM and ExtraTrees on official labels, then fixed sequence alignment",
                "validation_limit": "Recorded settings were selected in prior experiments. OOF diagnostics are not a new independent holdout; sequence references exclude the target subject, not every subject in its classifier fold."}
    write_json(output / "stage_protocol.json", protocol)
    train, test, labels, users, train_meta, test_meta = validate_observations(feature_path, train_meta_path, test_meta_path)
    oof_lgb, oof_et = np.zeros((len(train), 40), np.float32), np.zeros((len(train), 40), np.float32)
    test_lgb, test_et = np.zeros((len(test), 40), np.float64), np.zeros((len(test), 40), np.float64)
    folds, seen, reports = np.full(len(train), -1, np.int8), np.zeros(len(train), np.int8), []
    with threadpool_limits(limits=threads):
        for fold, held in enumerate(FOLDS):
            validation = np.isin(users, held)
            fitting = ~validation
            if set(labels[fitting]) != set(range(40)):
                raise ValueError("A fitting fold is missing classes")
            lgb = LGBMClassifier(objective="multiclass", num_class=40, n_estimators=180,
                learning_rate=0.06, num_leaves=23, min_child_samples=12, max_bin=63,
                subsample=0.90, subsample_freq=1, colsample_bytree=0.50, reg_alpha=0.05,
                reg_lambda=1.5, random_state=20260801 + fold, n_jobs=threads, verbosity=-1)
            et = ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", random_state=20260811 + fold, n_jobs=threads)
            lgb.fit(train[fitting], labels[fitting])
            et.fit(train[fitting], labels[fitting])
            if not np.array_equal(lgb.classes_, np.arange(40)) or not np.array_equal(et.classes_, np.arange(40)):
                raise RuntimeError("Teacher probability class order mismatch")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                oof_lgb[validation] = lgb.predict_proba(train[validation]).astype(np.float32)
                oof_et[validation] = et.predict_proba(train[validation]).astype(np.float32)
                test_lgb += lgb.predict_proba(test) / len(FOLDS)
                test_et += et.predict_proba(test) / len(FOLDS)
            folds[validation] = fold
            seen[validation] += 1
            record = {"fold": fold, "held_users": held, "fit_users": sorted(map(int, set(users[fitting]))),
                      "fit_rows": int(fitting.sum()), "validation_rows": int(validation.sum()),
                      "lgb_correct": int((oof_lgb[validation].argmax(1) == labels[validation]).sum()),
                      "extra_trees_correct": int((oof_et[validation].argmax(1) == labels[validation]).sum()),
                      "elapsed_seconds": time.perf_counter() - started}
            if save_models:
                model_dir = output / "fold_models"
                model_dir.mkdir(exist_ok=True)
                path = model_dir / f"fold_{fold}.joblib"
                joblib.dump({"lgb": lgb, "extra_trees": et, "held_users": held,
                             "fit_users": record["fit_users"]}, path, compress=3)
                record["model"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            reports.append(record)
            write_json(output / "fold_progress.json", reports)
            print(json.dumps(record), flush=True)
            del lgb, et
            gc.collect()
    if not np.all(seen == 1):
        raise RuntimeError("Every training row must receive exactly one held-fold prediction")
    train_sensor = 0.575 * oof_lgb.astype(np.float64) + 0.425 * oof_et.astype(np.float64)
    # The recorded producer serialized averaged components to float32 before blending.
    test_sensor = 0.575 * test_lgb.astype(np.float32).astype(np.float64) + 0.425 * test_et.astype(np.float32).astype(np.float64)
    train_sessions = sequence.build_sessions(train_meta, RECIPE["gap_seconds"], True)
    test_sessions = sequence.build_sessions(test_meta, RECIPE["gap_seconds"], False)
    train_sequence, train_choices = sequence.sequence_probabilities(train_sensor, labels,
        {"reference": train_sessions, "target": train_sessions}, RECIPE["hold_penalty"], RECIPE["skip_penalty"], RECIPE["top_k"], test=False)
    test_sequence, test_choices = sequence.sequence_probabilities(test_sensor, labels,
        {"reference": train_sessions, "target": test_sessions}, RECIPE["hold_penalty"], RECIPE["skip_penalty"], RECIPE["top_k"], test=True)
    arrays = dict(train_sensor=train_sensor, test_sensor=test_sensor,
                  train_sequence=train_sequence, test_sequence=test_sequence)
    for name, values in arrays.items():
        if not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError(f"Invalid probabilities: {name}")
        np.testing.assert_allclose(values.sum(1), 1.0, atol=1e-6, rtol=0)
    array_path = output / "teacher_arrays.npz"
    temporary = output / "teacher_arrays.tmp.npz"
    np.savez_compressed(temporary, **arrays, labels=labels, users=users, folds=folds,
                        train_groups=train_meta.group.astype(str).to_numpy(dtype=str),
                        test_groups=test_meta.group.astype(str).to_numpy(dtype=str))
    os.replace(temporary, array_path)
    manifest = {**protocol, "status": "completed", "array_file": array_path.name,
                "array_sha256": sha256(array_path), "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                "fold_results": reports, "model_serialization_requested": bool(save_models),
                "base_oof_correct": int((train_sensor.argmax(1) == labels).sum()),
                "sequence_sessions": {"train": len(train_sessions), "test": len(test_sessions)},
                "sequence_targets_aligned": {"train": len(train_choices), "test": len(test_choices)},
                "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "stage_manifest.json", manifest)
    print(json.dumps({"stage": "sensor", "status": "completed", "array_file": str(array_path),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return array_path, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--train-meta", required=True)
    parser.add_argument("--test-meta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_sensor_stage(args.features, args.train_meta, args.test_meta, args.out, args.save_models, args.threads)
