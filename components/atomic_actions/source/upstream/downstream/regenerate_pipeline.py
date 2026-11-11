"""Run downstream CPU fitting only from all required regenerated stage outputs.

Six explicit upstream stage directories are verified first. The seventh stage,
fusion, is computed from the regenerated sensor, kinematic, and raw-adaptive
arrays. The shared assembler verifies all seven again before asset creation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time


REQUIRED_STAGES = {"native", "sensor", "kinematic", "raw_adaptive", "skateformer", "efficientgcn"}
EXPECTED_RUNTIME = {"python": "3.12.14", "numpy": "2.4.4", "scipy": "1.18.0",
                    "scikit-learn": "1.9.0", "torch": "2.10.0+cpu"}
# The completed private downstream notebook also verified this exact Linux
# runtime. Both profiles keep every numerical package fixed; no general
# version-range fallback is accepted.
SUPPORTED_RUNTIMES = (EXPECTED_RUNTIME, {**EXPECTED_RUNTIME, "python": "3.12.13"})


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fact(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_stages(items):
    result = {}
    for item in items:
        pair = item.split("=", 1)
        if len(pair) != 2 or not pair[0] or not pair[1] or pair[0] in result:
            raise ValueError("Supply each stage exactly once as NAME=DIRECTORY")
        result[pair[0]] = Path(pair[1]).resolve()
    if set(result) != REQUIRED_STAGES:
        raise ValueError("Require exactly six upstream stages; fusion is freshly generated. "
                         f"Missing={sorted(REQUIRED_STAGES-set(result))}; extra={sorted(set(result)-REQUIRED_STAGES)}")
    return result


def check_runtime():
    actual = {"python": platform.python_version()}
    actual.update({p: importlib.metadata.version(p) for p in EXPECTED_RUNTIME if p != "python"})
    if actual not in SUPPORTED_RUNTIMES:
        raise RuntimeError("A verified downstream runtime is required: " + json.dumps({"supported": SUPPORTED_RUNTIMES, "actual": actual}))
    return actual


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_inputs(stages, train_metadata, test_metadata, template, observation_cache, skeleton_test_metadata):
    """Read and validate current inputs without writing or fitting anything."""
    import numpy as np
    import torch
    import assemble_teacher_inputs as assembler
    import dst_ridge_core_v2 as core

    train, test, sample = map(read_rows, (train_metadata, test_metadata, template))
    skeleton_rows = read_rows(skeleton_test_metadata)
    if len(train) != 2931 or len(test) != 405 or len(sample) != 405 or len(skeleton_rows) != 405:
        raise ValueError("Official metadata/template population mismatch")
    train_groups = np.asarray([r["group"] for r in train], dtype=str)
    test_groups = np.asarray([r["group"] for r in test], dtype=str)
    skeleton_groups = np.asarray([r["group"] for r in skeleton_rows], dtype=str)
    if not np.array_equal(skeleton_groups, test_groups):
        raise ValueError("The student observation cache must already follow canonical test row order")
    paths = [f"small_model_track_test/{g}/" for g in test_groups]
    if [r["path"] for r in sample] != paths:
        raise ValueError("Official template paths disagree with test metadata")
    labels = np.asarray([int(r["label"]) for r in train], np.int64)
    users = np.asarray([int(r["user"]) for r in train], np.int64)
    receipts = []
    for name in sorted(REQUIRED_STAGES):
        arrays, receipt = assembler.read_stage(name, stages[name], train_groups, test_groups, labels, users)
        receipts.append(receipt)
        del arrays
    native_manifest = json.loads((stages["native"] / "stage_manifest.json").read_text(encoding="utf-8-sig"))
    observation_sha = sha(observation_cache)
    native_observation_inputs = [r for r in native_manifest["inputs"] if r.get("role") == "official_skeleton_observations"]
    if not any(r["sha256"] == observation_sha for r in native_observation_inputs):
        raise ValueError("Student observations must match the native stage's verified official skeleton input")
    with np.load(observation_cache, allow_pickle=False) as archive:
        if set(archive.files) != {"train", "test", "train_n_valid", "test_n_valid"}:
            raise ValueError("An official observation-only four-array cache is required")
        sequences, counts = archive["test"], archive["test_n_valid"]
    if (sequences.shape != (405, 128, 17, 3) or sequences.dtype != np.float32 or counts.shape != (405,)
            or not np.isfinite(sequences).all() or np.any((counts < 1) | (counts > 128))
            or not np.array_equal(counts, np.asarray([int(r["n_valid"]) for r in skeleton_rows]))):
        raise ValueError("Official student observation shape, values, or row counts differ")
    learned = native_manifest["learned_inputs_used"]
    if len(learned) != 1 or learned[0].get("role") != "native_checkpoint":
        raise ValueError("The native stage must identify its single encoder checkpoint")
    checkpoint = Path(learned[0]["path"])
    if not checkpoint.is_absolute():
        checkpoint = stages["native"] / checkpoint
    if sha(checkpoint) != learned[0]["sha256"]:
        raise ValueError("Native learned input fingerprint mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Native stage checkpoint must contain the recorded tensor model")
    native_state = {k.removeprefix("module."): v for k, v in payload["model"].items()}
    if core.state_dict_sha256(native_state) != core.EXPECTED_DST_STATE_SHA256:
        raise ValueError("Native stage and compact student encoder states disagree")
    observation = {"file": fact(observation_cache), "test_metadata": fact(skeleton_test_metadata),
        "test_tensor_sha256": core.array_sha256(sequences),
        "test_counts_int64_sha256": core.array_sha256(counts.astype(np.int64)), "test_rows": len(sequences)}
    return receipts, native_state, {"checkpoint": fact(checkpoint),
        "canonical_tensor_sha256": core.EXPECTED_DST_STATE_SHA256}, observation


def run(stages, train_metadata, test_metadata, template, observation_cache, skeleton_test_metadata,
        output_root, *, validate_only=False):
    if set(stages) != REQUIRED_STAGES:
        raise ValueError("All six regenerated upstream stages are mandatory")
    runtime = check_runtime()
    output = Path(output_root).resolve()
    # The original local integration wrote only to E:. On a declared host any drive is
    # allowed; the check is retained otherwise.
    if (os.name == "nt" and output.drive.upper() != "E:"
            and os.environ.get("CUHKX_ALLOW_NON_LINUX_HOST") != "1"):
        raise ValueError("This local integration writes only to E:; choose an E: output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output root to preserve prior results")
    started = time.perf_counter()
    receipts, native_state, native_receipt, observation_receipt = verify_inputs(
        stages, train_metadata, test_metadata, template, observation_cache, skeleton_test_metadata)
    if validate_only:
        result = {"status": "all_six_upstream_inputs_verified_no_execution", "runtime": runtime,
                  "stages": receipts, "native_encoder": native_receipt, "observation_cache": observation_receipt,
                  "fusion_executed": False, "downstream_training_executed": False, "files_written": False}
        print(json.dumps(result, indent=2), flush=True)
        return result
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "temporary"
    temporary.mkdir()
    for name in ("TEMP", "TMP", "TMPDIR"):
        os.environ[name] = str(temporary)
    import torch
    from threadpoolctl import threadpool_limits
    import assemble_teacher_inputs as assembler
    import fusion_stage
    import run_pipeline

    with threadpool_limits(limits=1):
        fusion = output / "fusion"
        fusion_stage.run_fusion(stages["sensor"], stages["kinematic"], stages["raw_adaptive"],
                               train_metadata, test_metadata, fusion)
        complete_stages = dict(stages, fusion=fusion)
        assets = output / "assets"
        assembly = assembler.assemble(complete_stages, train_metadata, test_metadata, template, assets)
        torch.save(native_state, assets / "native_state.pt")
        provenance = {"status": "new_assets_from_all_seven_verified_regenerated_stages",
            "upstream_stage_receipts": assembly["stage_receipts"], "native_encoder": native_receipt,
            "student_observations": observation_receipt, "runtime": runtime,
            "wrapper_source": fact(__file__), "assembly_report": fact(assets / "assembly_report.json"),
            "learned_inputs": [r["learned_inputs_used"] for r in assembly["stage_receipts"]],
            "training_scope": "Fixed downstream models and compact pseudo-label head are fitted here. Upstream fitting versus checkpoint inference is specified in each stage manifest."}
        write_json(assets / "input_provenance.json", provenance)
        asset_names = ("teacher_inputs.npz", "train_sensor_metadata.csv", "test_sensor_metadata.csv",
                       "native_state.pt", "input_provenance.json", "assembly_report.json")
        manifest = {name: {"bytes": (assets / name).stat().st_size, "sha256": sha(assets / name)} for name in asset_names}
        write_json(assets / "asset_manifest.json", manifest)
        downstream = output / "downstream"
        result = run_pipeline.run(assets, template, downstream, observation_cache=observation_cache,
                                  provenance_path=assets / "input_provenance.json")
    report = {"status": "regenerated_pipeline_complete", "stage_count": len(complete_stages),
        "assembly": assembly, "downstream": result, "runtime": runtime,
        "source_files": [fact(p) for p in sorted(Path(__file__).parent.glob("*.py"))],
        "seconds": time.perf_counter() - started, "gpu_used": False,
        "output_files": {name: fact(downstream / name) for name in
                         ("submission.csv", "skeleton_model.pt", "run_report.json")}}
    write_json(output / "regenerated_pipeline_report.json", report)
    print(json.dumps({"status": report["status"], "output_root": str(output),
                      "seconds": report["seconds"], "csv_sha256": report["output_files"]["submission.csv"]["sha256"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", default=[], metavar="NAME=DIRECTORY")
    for name in ("train-metadata", "test-metadata", "template", "observation-cache", "skeleton-test-metadata", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = vars(parser.parse_args())
    try:
        args["stages"] = parse_stages(args.pop("stage"))
    except ValueError as error:
        parser.error(str(error))
    run(**args)
