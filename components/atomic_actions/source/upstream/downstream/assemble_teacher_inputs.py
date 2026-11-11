"""Assemble regenerated teacher outputs in the official metadata row order.

This step performs no fitting or output-based selection. Every stage is required;
there is no fallback to a combined teacher cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


FOLDS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (16, 17, 18),
         (19, 20, 21), (22, 23, 24))
STAGES = {
    "native": {"train_anchor": 40, "test_anchor": 40,
               "train_features": 512, "test_features": 512},
    "sensor": {"train_sensor": 40, "test_sensor": 40,
               "train_sequence": 40, "test_sequence": 40},
    "kinematic": {"train_kinematic": 40, "test_kinematic": 40},
    "raw_adaptive": {"train_raw_logits": 40, "test_raw_logits": 40},
    "skateformer": {"train_skate": 40, "test_skate": 40},
    "efficientgcn": {"train_egcn": 40, "test_egcn": 40},
    "fusion": {"train_fusion": 40, "test_fusion": 40},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def align_order(actual, expected):
    """Return a permutation, rejecting duplicate, missing and unexpected rows."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.ndim != 1 or expected.ndim != 1:
        raise ValueError("Row identifiers must be one-dimensional")
    if actual.dtype.kind != "U" or expected.dtype.kind != "U":
        raise ValueError("Row identifiers must be Unicode arrays")
    index = {name: i for i, name in enumerate(actual.tolist())}
    if len(index) != len(actual) or len(set(expected.tolist())) != len(expected):
        raise ValueError("Duplicate row identifiers")
    if set(index) != set(expected.tolist()):
        raise ValueError("Missing or unexpected row identifiers")
    return np.array([index[name] for name in expected], dtype=np.int64)


def validate_stage(name, archive, train_groups, test_groups, labels, users):
    train_order = align_order(archive["train_groups"], train_groups)
    test_order = align_order(archive["test_groups"], test_groups)
    for key, expected in (("labels", labels), ("users", users)):
        values = archive[key]
        if values.shape != expected.shape or values.dtype.kind not in "iu":
            raise ValueError(f"{name}: invalid {key} shape or dtype")
        if not np.array_equal(values[train_order], expected):
            raise ValueError(f"{name}: {key} disagree with official metadata")
    arrays = {}
    for key, width in STAGES[name].items():
        order = train_order if key.startswith("train_") else test_order
        values = archive[key]
        if values.shape != (len(order), width) or values.dtype.kind != "f":
            raise ValueError(f"{name}: invalid {key} shape or dtype")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: nonfinite values in {key}")
        if not key.endswith(("features", "logits")):
            if np.any(values < 0) or np.any(values.sum(axis=1) <= 0):
                raise ValueError(f"{name}: invalid probability rows in {key}")
        # Preserve precision and serialization boundaries from the producing stage.
        arrays[key] = values[order]
    return arrays


def read_stage(name, directory, train_groups, test_groups, labels, users):
    directory = Path(directory)
    manifest_path = directory / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("stage") != name:
        raise ValueError(f"Wrong stage in {manifest_path}")
    array_path = (directory / manifest["array_file"]).resolve()
    if not array_path.is_relative_to(directory.resolve()):
        raise ValueError("Stage output must be inside its stage directory")
    if sha(array_path) != manifest["array_sha256"]:
        raise ValueError(f"Changed stage output: {array_path}")
    if not manifest.get("inputs") or not manifest.get("source_files"):
        raise ValueError(f"{name}: input and source receipts are required")
    if "learned_inputs_used" not in manifest:
        raise ValueError(f"{name}: learned input disclosure is required")
    for item in manifest["inputs"] + manifest["source_files"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = directory / path
        if sha(path) != item["sha256"]:
            raise ValueError(f"Changed recorded source/input: {path}")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = validate_stage(name, archive, train_groups, test_groups, labels, users)
    return arrays, {"stage": name, "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": sha(manifest_path),
                    "array_sha256": manifest["array_sha256"],
                    "method": manifest.get("method"),
                    "learned_inputs_used": manifest["learned_inputs_used"]}


def assemble(stages, train_metadata, test_metadata, template, output):
    if set(stages) != set(STAGES):
        missing = sorted(set(STAGES) - set(stages))
        extra = sorted(set(stages) - set(STAGES))
        raise ValueError(f"All regenerated stages required. Missing={missing}; extra={extra}")
    train, test = read_metadata(train_metadata), read_metadata(test_metadata)
    sample = read_metadata(template)
    if len(train) != 2931 or len(test) != 405 or len(sample) != len(test):
        raise ValueError("Unexpected official train/test/template row counts")
    train_groups = np.asarray([r["group"] for r in train], dtype=str)
    test_groups = np.asarray([r["group"] for r in test], dtype=str)
    align_order(train_groups, train_groups)
    align_order(test_groups, test_groups)
    expected_paths = [f"small_model_track_test/{g}/" for g in test_groups]
    if [r["path"] for r in sample] != expected_paths:
        raise ValueError("Official test metadata and sample template order disagree")
    labels = np.asarray([int(r["label"]) for r in train], dtype=np.int64)
    users = np.asarray([int(r["user"]) for r in train], dtype=np.int64)
    path_labels = np.asarray([int(g.split("_", 1)[0]) for g in train_groups])
    if not np.array_equal(labels, path_labels) or not np.array_equal(np.unique(labels), np.arange(40)):
        raise ValueError("Labels do not match the official training paths/classes")
    fold_map = {user: fold for fold, held in enumerate(FOLDS) for user in held}
    if not set(users.tolist()).issubset(fold_map):
        raise ValueError("Unexpected training subjects")
    arrays = {"labels": labels, "users": users,
              "folds": np.asarray([fold_map[int(u)] for u in users], dtype=np.int8)}
    receipts = []
    # Validate every dependency before creating a downstream input package.
    for name in STAGES:
        values, receipt = read_stage(name, stages[name], train_groups, test_groups, labels, users)
        arrays.update(values)
        receipts.append(receipt)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / "teacher_inputs.npz"
    if array_path.exists():
        raise FileExistsError(f"Use a new output directory: {array_path}")
    temporary = output / "teacher_inputs.partial.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(array_path)
    for source, name in ((train_metadata, "train_sensor_metadata.csv"),
                         (test_metadata, "test_sensor_metadata.csv")):
        shutil.copyfile(source, output / name)
    report = {"assembly_status": "all_required_regenerated_stages_assembled",
              "array_file": array_path.name, "array_sha256": sha(array_path),
              "source_file": str(Path(__file__).resolve()), "source_sha256": sha(__file__),
              "rows": {"train": len(train), "test": len(test)}, "stage_receipts": receipts,
              "metadata": [{"path": str(Path(p).resolve()), "sha256": sha(p)}
                           for p in (train_metadata, test_metadata, template)],
              "historical_reference_used": False,
              "competition_compliance": "not_assessed_by_assembly",
              "full_upstream_training": "not_proven_by_assembly"}
    (output / "assembly_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pairs = [value.split("=", 1) for value in args.stage]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        parser.error("Provide each stage once as NAME=DIRECTORY")
    print(json.dumps(assemble(dict(pairs), args.train_metadata, args.test_metadata,
                              args.template, args.out), indent=2))
