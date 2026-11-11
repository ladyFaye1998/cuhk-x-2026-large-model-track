"""CPU-only preparation of official observations for the original raw-adaptive model.

No model initialization, prediction input or accelerator allocation occurs here.
The archive stores and decoder are the original functions in raw_adaptive_vendor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

import raw_adaptive_vendor as core


SCHEMA = "cuhkx-original-raw-adaptive-observations/v1"
ARRAY_KEYS = {"train_groups", "test_groups", "train_raw", "test_raw", "train_valid", "test_valid", "labels", "users"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def plan_raw(files):
    """Calculate exact original byte ranges without requesting network data."""
    train = [r for r in core.read_index(files["train_index"]) if core.is_skeleton(r)]
    test = [r for r in core.read_index(files["test_index"]) if core.is_skeleton(r)]
    if not train or not test or not {r["disk"] for r in train}.issubset(core.TRAIN_DISK_IDS):
        raise ValueError("Empty or unexpected official Skeleton archive index")
    spans = []
    for disk in sorted({r["disk"] for r in train}):
        rows = [r for r in train if r["disk"] == disk]
        start = min(r["offset"] for r in rows)
        end = max(r["offset"] + r["compressed"] + 1024 for r in rows) - 1
        spans.append(dict(disk=disk, file_id=core.TRAIN_DISK_IDS[disk], start=start,
                          end=end, bytes=end-start+1, entries=len(rows)))
    ranges = [dict(file_id=core.TEST_ID, start=a, end=b, bytes=b-a+1, entries=len(rows))
              for a, b, rows in core.merged_intervals(test)]
    train_groups = sorted({core.train_group(r["name"]) for r in train}, key=core.numeric_key)
    test_groups = sorted({core.test_group(r["name"]) for r in test}, key=core.numeric_key)
    tensor_bytes = (len(train_groups)+len(test_groups))*core.N_FRAMES*core.N_JOINTS*core.N_CHANNELS*4
    temporary_bytes = sum(r["bytes"] for r in spans)
    return dict(schema=SCHEMA, train_spans=spans, test_ranges=ranges,
                total_download_bytes=temporary_bytes+sum(r["bytes"] for r in ranges),
                temporary_disk_bytes=temporary_bytes,
                uncompressed_tensor_bytes=tensor_bytes,
                recommended_free_disk_bytes=max(temporary_bytes, tensor_bytes)+64*1024*1024,
                train_rows=len(train_groups), test_rows=len(test_groups),
                decoder_source_sha256=sha256(core.__file__))


def validate_observations(arrays, files, receipt):
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Unexpected observation cache arrays; only declared observations and row metadata are permitted")
    for split, count in (("train", 2931), ("test", 405)):
        raw, valid, groups = (arrays[f"{split}_{key}"] for key in ("raw", "valid", "groups"))
        if raw.dtype != np.float32 or raw.shape != (count, 96, 17, 8) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid {split} observation tensor")
        if valid.dtype != np.float32 or valid.shape != (count,) or not np.all(valid == 1):
            raise ValueError(f"Incomplete {split} observations")
        if groups.dtype.kind != "U" or groups.shape != (count,) or len(set(groups)) != count:
            raise ValueError(f"Invalid {split} row IDs")
        if hashlib.sha256(raw.tobytes()).hexdigest() != receipt["tensor_sha256"][split]:
            raise ValueError(f"Decoded {split} tensor checksum mismatch")
    import re
    train_expected = sorted({core.train_group(r["name"]) for r in core.read_index(files["train_index"]) if core.is_skeleton(r)}, key=core.numeric_key)
    test_expected = sorted({core.test_group(r["name"]) for r in core.read_index(files["test_index"]) if core.is_skeleton(r)}, key=core.numeric_key)
    paths = pd.read_csv(files["template"], usecols=["path"]).path.astype(str)
    if arrays["train_groups"].tolist() != train_expected or arrays["test_groups"].tolist() != test_expected:
        raise ValueError("Observation row IDs differ from the official archive directory")
    if [p.strip("/").rsplit("/", 1)[-1] for p in paths] != test_expected:
        raise ValueError("Official sample template order mismatch")
    labels = np.asarray([int(g.split("/", 1)[0].split("_", 1)[0]) for g in train_expected], np.int64)
    users = np.asarray([int(re.search(r"user(\d+)", g).group(1)) for g in train_expected], np.int64)
    if not np.array_equal(arrays["labels"], labels) or not np.array_equal(arrays["users"], users):
        raise ValueError("Training labels/users do not match official observation paths")
    if arrays["labels"].dtype != np.int64 or arrays["users"].dtype != np.int64:
        raise ValueError("Training label/user dtype mismatch")


def run_preparation(input_root, out_dir, transport="official-mirror"):
    """Download/decode on CPU and persist a checksum-verified observation-only cache."""
    from raw_adaptive_stage import input_files, prepare_raw
    output = Path(out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; existing artifacts are preserved")
    files = input_files(input_root)
    plan = plan_raw(files)
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    if free < plan["recommended_free_disk_bytes"]:
        raise RuntimeError(f"Insufficient output disk: {free} available, {plan['recommended_free_disk_bytes']} required")
    started = time.time()
    write_json(output / "download_plan.json", plan)
    print(json.dumps({"stage": "raw_adaptive_cpu_preparation", "plan": plan}), flush=True)
    if transport not in ("original", "official-mirror"):
        raise ValueError("Unknown raw archive transport")
    original_fetch = core.fetch_range
    transport_module = None
    try:
        if transport == "official-mirror":
            import raw_adaptive_transport as transport_module
            core.fetch_range = transport_module.fetch_range
        result = prepare_raw(files, output)
    finally:
        core.fetch_range = original_fetch
    keys = ("train_groups", "test_groups", "train_raw", "test_raw", "train_valid", "test_valid", "labels", "users")
    arrays = {key: np.asarray(value, dtype=str) if key.endswith("groups") else value
              for key, value in zip(keys, result)}
    receipt = json.loads((output / "raw_observation_receipt.json").read_text())
    validate_observations(arrays, files, receipt)
    temporary = output / "raw_observations.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    array_path = output / "raw_observations.npz"
    os.replace(temporary, array_path)
    inputs = []
    copied_root = output / "official_indexes"
    copied_root.mkdir()
    for role, path in files.items():
        target = copied_root / path.name
        shutil.copyfile(path, target)
        inputs.append(dict(role=role, path=str(path.resolve()), file=f"official_indexes/{path.name}", sha256=sha256(path)))
    manifest = dict(schema=SCHEMA, stage="raw_adaptive_observation_preparation", status="completed",
                    learned_inputs_used=[], accelerator_used=False,
                    array_file=array_path.name, array_sha256=sha256(array_path), array_bytes=array_path.stat().st_size,
                    array_shapes={key: list(value.shape) for key, value in arrays.items()},
                    decoder_source_sha256=sha256(core.__file__), original_source_sha256=core.ORIGINAL_SOURCE_SHA256,
                    source_files=[dict(path=str(Path(p).resolve()), sha256=sha256(p)) for p in (__file__, core.__file__, __import__("raw_adaptive_stage").__file__)],
                    inputs=inputs, receipt_file="raw_observation_receipt.json",
                    receipt_sha256=sha256(output / "raw_observation_receipt.json"),
                    elapsed_seconds=time.time()-started,
                    meaning="Official observations decoded with the original deterministic body tracking/resampling; no learned arrays, states or predictions")
    manifest["transport"] = transport_module.receipt() if transport_module else {"method": "Unmodified original public Drive fetch", "decoder_changed": False}
    if transport_module:
        manifest["source_files"].append(dict(path=str(Path(transport_module.__file__).resolve()), sha256=sha256(transport_module.__file__)))
    write_json(output / "preparation_manifest.json", manifest)
    print(json.dumps({"stage": manifest["stage"], "status": "completed", "array_bytes": manifest["array_bytes"], "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return array_path, manifest


def load_prepared(prepared_root, files):
    """Verify the saved CPU preparation against current official indexes and decoder."""
    root = Path(prepared_root).resolve(strict=True)
    manifest = json.loads((root / "preparation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "completed" or manifest.get("learned_inputs_used") != []:
        raise ValueError("Unexpected or incomplete observation preparation manifest")
    if manifest.get("decoder_source_sha256") != sha256(core.__file__):
        raise ValueError("Cached observations were not prepared by this exact original decoder source")
    if manifest.get("array_file") != "raw_observations.npz" or manifest.get("receipt_file") != "raw_observation_receipt.json":
        raise ValueError("Unexpected prepared observation filenames")
    expected = {row["role"]: row["sha256"] for row in manifest["inputs"]}
    if expected != {role: sha256(path) for role, path in files.items()}:
        raise ValueError("Cached observations use different official indexes/template")
    array_path, receipt_path = root / manifest["array_file"], root / manifest["receipt_file"]
    if sha256(array_path) != manifest["array_sha256"] or sha256(receipt_path) != manifest["receipt_sha256"]:
        raise ValueError("Prepared observation file checksum mismatch")
    with np.load(array_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_observations(arrays, files, receipt)
    result = tuple(arrays[key].tolist() if key.endswith("groups") else arrays[key]
                   for key in ("train_groups", "test_groups", "train_raw", "test_raw", "train_valid", "test_valid", "labels", "users"))
    return result, manifest, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out", help="Empty CPU preparation output directory")
    parser.add_argument("--plan-only", action="store_true", help="Print exact range/disk totals without downloading anything")
    parser.add_argument("--transport", choices=("original", "official-mirror"), default="official-mirror")
    args = parser.parse_args()
    if args.plan_only:
        from raw_adaptive_stage import input_files
        print(json.dumps(plan_raw(input_files(args.input_root)), indent=2))
    elif not args.out:
        parser.error("--out is required unless --plan-only")
    else:
        run_preparation(args.input_root, args.out, args.transport)
