#!/usr/bin/env python3
"""Frozen full-fit HAU inference for an independently validated model voter.

The model and hard-agreement deployment rule were frozen using training-only
held-user OOF predictions before this script accessed anonymous media.  This
script emits model predictions and label-free features only; it cannot create
or submit a competition candidate.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np


SEED = 20260822
TRAIN_QA_SHA256 = "2509ed00f9305d552378618d8987559bdff7a4b56241c630ba99dc4051f535bc"
TEST_QA_SHA256 = "d694c7abc5a003d5c9048098880f0f77716fae0eb18d1c9fe9330e4e987320a5"
HAU_ARCHIVE_BYTES = 3_674_779_437
EXPECTED_PATHS = 809
EXPECTED_QA = 3525
EXPECTED_TEST_PATHS = 144
EXPECTED_TEST_QA = 610
EXPECTED_TEST_DEPTH_PROXY_UNITS = (
    "large_model_track_test/LM_test_0103",
    "large_model_track_test/LM_test_0104",
    "large_model_track_test/LM_test_0105",
    "large_model_track_test/LM_test_0106",
)
FOLD_USERS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (16, 17, 18), (19, 20, 21), (22, 23, 24))
WINDOWS = 12
FRAMES = 16
SOURCE_SIZE = 128
VJEPA_SIZE = 256
VJEPA_RESIZE = 292
WINDOW_HALF_SPAN = 0.05
MODEL_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
MODEL_REVISION = "4aa02df83918538fc21cfaf576382fa20e489a80"
MODEL_FILES = {
    "README.md": (3712, "400bbc19de13b62f092738b668b88aa026071bc4fc9075faab729638d9850ff1"),
    "config.json": (14953, "9ea266370f726ba85ba03080abab9632233d0742a084f0106d90d909772e4aab"),
    "model.safetensors": (1_502_018_592, "35aeab96e8513d9cc60ee08c1642dd121c86452c926ce3599b3317896a320396"),
    "video_preprocessor_config.json": (1464, "6a5687451deda1331cb4c0ba08fc440a21e285a8905522e3537004eddfca13bb"),
}
SIGLIP_REPO = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
LETTERS = "ABCD"
ACTIONS = (
    "Brushing teeth", "Calling", "Checking body temperature", "Checking the time",
    "Combing hair", "Drinking", "Eating", "Folding clothes", "Getting dressed",
    "Grabbing utensils", "Jumping jacks", "Listening to the music with headphones",
    "Lunges", "Lying down", "Massaging oneself", "Mopping", "Peeling fruit",
    "Playing games", "Pouring", "Reading", "Running", "Sitting down", "Squats",
    "Standing up", "Stirring", "Stretching", "Sweeping", "Taking a selfie",
    "Taking medicine", "Turning a page", "Typing on a keyboard", "Undressing",
    "Using a phone", "Walking", "Washing dishes", "Washing face", "Watching TV",
    "Wiping hands", "Wiping surface", "Writing",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
PROTOCOL = {
    "id": "CUHKX_LARGE_HAU_VJEPA2L_SIGLIP_FULLFIT_INFERENCE_V1",
    "status": "FROZEN_AFTER_TRAIN_ONLY_GATE_BEFORE_ANONYMOUS_INFERENCE",
    "created": "2026-08-22",
    "scope": "official HAU training paths and training answers only",
    "folds": [list(values) for values in FOLD_USERS],
    "features": {
        "vjepa2": "12 local windows x (Depth_Color, IR), frozen pooled tokens and SSV2 logits",
        "siglip": "12 IR center frames plus frozen action/emotion text embeddings",
        "pace": "label-free temporal-difference and luminance summaries",
    },
    "head": "fixed 192-d two-layer temporal transformer; action presence, occurrence time, option-listwise and emotion-text losses",
    "epochs": 220,
    "optimizer": "AdamW(lr=0.002,weight_decay=0.03), cosine, full fold batch",
    "train_only_gate": {
        "rule": "exact answer agreement with frozen SigLIP triplet voter against parent",
        "changes": 118, "wins": 109, "losses": 7, "net": 102,
        "fold_net": [21, 14, 17, 22, 11, 17], "negative_users": 0,
    },
    "fullfit": {"seeds": 6, "epochs_per_seed": 220, "ensemble": "mean logits and occurrence times"},
    "anonymous_media_coverage": {
        "rule": "Use paired Depth_Color and IR when present; for the four relay units with absent Depth_Color, duplicate the available IR stream into the depth branch.",
        "depth_proxy_units": list(EXPECTED_TEST_DEPTH_PROXY_UNITS),
        "frozen_after_coverage_error_before_any_anonymous_prediction": True,
    },
    "authorization": {"test_model_inference": True, "candidate": False, "submission": False},
    "prohibitions": ["test labels", "manual test inspection", "aim/reference histograms", "leaderboard inference", "candidate writing"],
}


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def install_runtime() -> dict[str, str]:
    # Kaggle can assign a Pascal P100 (sm_60). Its current base image ships a
    # torch wheel whose embedded kernels begin at sm_70, so pin the official
    # CUDA 12.6 family that still includes Pascal kernels.
    torch_required = {"torch": "2.7.1", "torchvision": "0.22.1", "torchaudio": "2.7.1"}
    required = {
        **torch_required,
        "transformers": "4.57.1",
        "huggingface_hub": "0.36.0",
        "safetensors": "0.6.2",
    }
    observed = {}
    for package in required:
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed[package] = "ABSENT"
    if any(observed[name] != version for name, version in torch_required.items()):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--quiet",
             "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
             "--index-url", "https://download.pytorch.org/whl/cu126"],
            check=True,
        )
    if any(observed[name] != required[name] for name in ("transformers", "huggingface_hub", "safetensors")):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--quiet",
             "transformers==4.57.1", "huggingface_hub==0.36.0", "safetensors==0.6.2"],
            check=True,
        )
    final = {package: importlib.metadata.version(package) for package in required}
    normalized = {
        package: (version.split("+", 1)[0] if package in torch_required else version)
        for package, version in final.items()
    }
    if normalized != required:
        raise ExperimentError(f"runtime dependency drift: {final}")
    return final


def find_exact(root: Path, filename: str, expected_sha: str | None = None, expected_bytes: int | None = None) -> Path:
    matches = []
    for path in root.rglob(filename):
        if path.is_symlink() or not path.is_file():
            continue
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            continue
        if expected_sha is not None and sha256(path) != expected_sha:
            continue
        matches.append(path.resolve())
    if len(matches) != 1:
        raise ExperimentError(f"expected one exact {filename}, got {matches}")
    return matches[0]


def parse_user(path: str) -> int:
    match = re.fullmatch(r"HAU/user(\d+)/[^/]+", path)
    if match is None:
        raise ExperimentError(f"invalid HAU training path: {path}")
    return int(match.group(1))


def load_qa(path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row.get("source") == "HAU"]
    if len(rows) != EXPECTED_QA or len({row["qa_id"] for row in rows}) != EXPECTED_QA:
        raise ExperimentError("HAU QA row contract failed")
    paths = sorted({row["path"] for row in rows})
    if len(paths) != EXPECTED_PATHS:
        raise ExperimentError("HAU path count drift")
    users = np.asarray([parse_user(value) for value in paths], dtype=np.int16)
    expected_users = sorted({user for fold in FOLD_USERS for user in fold})
    if sorted(set(users.tolist())) != expected_users:
        raise ExperimentError("HAU user contract drift")
    index = {value: i for i, value in enumerate(paths)}
    categories = {name: 0 for name in ("single", "multi", "combination", "sequence", "emotion")}
    for row in rows:
        if row["path"] not in index or row["category"] not in categories:
            raise ExperimentError("QA alignment/category drift")
        categories[row["category"]] += 1
        if row["answer"] == "" or any(row[letter] == "" for letter in LETTERS):
            raise ExperimentError("empty training answer/option")
    if categories != {"single": 809, "multi": 809, "combination": 790, "sequence": 308, "emotion": 809}:
        raise ExperimentError(f"category counts drift: {categories}")
    return rows, paths, users, {"category_counts": categories, "paths_sha256": canonical_sha(paths)}


def parse_actions(value: str) -> list[int]:
    names = [part.strip() for part in value.split(",")]
    if not names or any(name not in ACTION_INDEX for name in names):
        raise ExperimentError(f"unknown action option: {value!r}")
    result = [ACTION_INDEX[name] for name in names]
    if len(set(result)) != len(result):
        raise ExperimentError(f"duplicate action in option: {value!r}")
    return result


def build_targets(rows: list[dict[str, str]], paths: list[str]) -> dict[str, Any]:
    n = len(paths)
    pindex = {path: i for i, path in enumerate(paths)}
    presence = np.zeros((n, 40), dtype=np.float32)
    known = np.zeros((n, 40), dtype=bool)
    single_opts = np.empty((n, 4), dtype=np.int64)
    single_ans = np.empty(n, dtype=np.int64)
    multi_opts = np.empty((n, 4), dtype=np.int64)
    multi_targets = np.zeros((n, 4), dtype=np.float32)
    combo_opts = np.zeros((n, 4, 40), dtype=np.float32)
    combo_ans = np.full(n, -1, dtype=np.int64)
    emotion_terms = sorted({row[letter] for row in rows if row["category"] == "emotion" for letter in LETTERS})
    emotion_index = {term: i for i, term in enumerate(emotion_terms)}
    emotion_opts = np.empty((n, 4), dtype=np.int64)
    emotion_ans = np.empty(n, dtype=np.int64)
    sequence_paths, sequence_order = [], []
    row_path_indices = np.asarray([pindex[row["path"]] for row in rows], dtype=np.int64)
    for row in rows:
        i = pindex[row["path"]]
        cat = row["category"]
        answer_letters = row["answer"]
        if any(letter not in LETTERS for letter in answer_letters) or len(set(answer_letters)) != len(answer_letters):
            raise ExperimentError(f"invalid answer grammar: {answer_letters}")
        if cat == "single":
            option_indices = [parse_actions(row[letter])[0] for letter in LETTERS]
            single_opts[i] = option_indices
            single_ans[i] = LETTERS.index(answer_letters)
            for j, action in enumerate(option_indices):
                known[i, action] = True
                presence[i, action] = float(LETTERS[j] in answer_letters)
        elif cat == "multi":
            option_indices = [parse_actions(row[letter])[0] for letter in LETTERS]
            multi_opts[i] = option_indices
            multi_targets[i] = [float(letter in answer_letters) for letter in LETTERS]
            for j, action in enumerate(option_indices):
                known[i, action] = True
                presence[i, action] = float(LETTERS[j] in answer_letters)
        elif cat == "combination":
            for j, letter in enumerate(LETTERS):
                combo_opts[i, j, parse_actions(row[letter])] = 1.0
            combo_ans[i] = LETTERS.index(answer_letters)
            exact = combo_opts[i, combo_ans[i]].astype(bool)
            known[i] = True
            presence[i] = exact.astype(np.float32)
        elif cat == "sequence":
            options = [parse_actions(row[letter])[0] for letter in LETTERS]
            if len(answer_letters) != 4:
                raise ExperimentError("sequence answer must be a permutation")
            sequence_paths.append(i)
            sequence_order.append([options[LETTERS.index(letter)] for letter in answer_letters])
        elif cat == "emotion":
            emotion_opts[i] = [emotion_index[row[letter]] for letter in LETTERS]
            emotion_ans[i] = LETTERS.index(answer_letters)
    if np.any(single_opts < 0) or np.any(multi_opts < 0) or np.any(emotion_opts < 0):
        raise ExperimentError("target construction failed")
    return {
        "presence": presence, "known": known,
        "single_opts": single_opts, "single_ans": single_ans,
        "multi_opts": multi_opts, "multi_targets": multi_targets,
        "combo_opts": combo_opts, "combo_ans": combo_ans,
        "emotion_terms": emotion_terms, "emotion_opts": emotion_opts, "emotion_ans": emotion_ans,
        "sequence_paths": np.asarray(sequence_paths, dtype=np.int64),
        "sequence_order": np.asarray(sequence_order, dtype=np.int64),
        "row_path_indices": row_path_indices,
    }


def build_member_maps(archive: zipfile.ZipFile, paths: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    wanted = set(paths)
    depth, infrared = {}, {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/").strip("/")
        modality = None
        suffix = None
        if name.endswith("/Depth_Color/Depth_Color.mp4"):
            modality, suffix = depth, "/Depth_Color/Depth_Color.mp4"
        elif name.endswith("/IR/IR.mp4"):
            modality, suffix = infrared, "/IR/IR.mp4"
        if modality is None:
            continue
        unit = name[: -len(suffix)]
        marker = unit.rfind("HAU/")
        if marker >= 0:
            unit = unit[marker:]
        if unit not in wanted:
            continue
        if unit in modality:
            raise ExperimentError(f"duplicate media member for {unit}")
        modality[unit] = info.filename
    if set(depth) != wanted or set(infrared) != wanted:
        raise ExperimentError(f"media coverage drift: depth={len(depth)}, ir={len(infrared)}")
    return depth, infrared


def center_square_resize(rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2
    height, width = rgb.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return cv2.resize(rgb[top:top + side, left:left + side], (size, size), interpolation=cv2.INTER_AREA)


def contrast_ir(rgb: np.ndarray) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    low, high = np.percentile(gray, [1.0, 99.0])
    if high > low + 1.0:
        gray = np.clip((gray.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def window_indices(total: int) -> np.ndarray:
    if total < 2:
        raise ExperimentError(f"invalid video frame count: {total}")
    centers = np.linspace(0.04, 0.96, WINDOWS)
    result = []
    for center in centers:
        start, end = max(0.0, center - WINDOW_HALF_SPAN), min(1.0, center + WINDOW_HALF_SPAN)
        result.append(np.linspace(start * (total - 1), end * (total - 1), FRAMES).round().astype(np.int32))
    return np.stack(result)


def decode_windows(path: Path, is_ir: bool) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = window_indices(total)
    requested = sorted({int(value) for value in indices.reshape(-1)})
    retained: dict[int, np.ndarray] = {}
    cursor = 0
    frame_index = 0
    while cursor < len(requested):
        ok, bgr = capture.read()
        if not ok or bgr is None:
            break
        if frame_index == requested[cursor]:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if is_ir:
                rgb = contrast_ir(rgb)
            retained[frame_index] = center_square_resize(rgb, SOURCE_SIZE)
            cursor += 1
        frame_index += 1
    capture.release()
    missing = sorted(set(requested) - set(retained))
    if missing:
        raise ExperimentError(f"decoder missed requested frames in {path.name}: {missing[:8]}")
    frames = np.stack([[retained[int(value)] for value in row] for row in indices]).astype(np.uint8)
    if frames.shape != (WINDOWS, FRAMES, SOURCE_SIZE, SOURCE_SIZE, 3):
        raise ExperimentError(f"decoded shape drift: {frames.shape}")
    gray = frames.astype(np.float32).mean(axis=-1) / 255.0
    diffs = np.abs(np.diff(gray, axis=1))
    pace = np.stack(
        [diffs.mean((1, 2, 3)), diffs.std((1, 2, 3)), np.quantile(diffs, .90, axis=(1, 2, 3)),
         gray.mean((1, 2, 3)), gray.std((1, 2, 3)),
         np.full(WINDOWS, math.log1p(total / fps) if fps > 0 else math.log1p(total), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    return frames, pace


def acquire_models(model_root: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoProcessor, VJEPA2ForVideoClassification

    vj_dir = model_root / "vjepa2"
    snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, allow_patterns=sorted(MODEL_FILES), local_dir=str(vj_dir))
    model_facts = {}
    for filename, (size, digest) in MODEL_FILES.items():
        path = vj_dir / filename
        fact = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if fact != {"bytes": size, "sha256": digest}:
            raise ExperimentError(f"V-JEPA2 model binding failed: {filename}: {fact}")
        model_facts[filename] = fact
    vjepa = VJEPA2ForVideoClassification.from_pretrained(
        str(vj_dir), local_files_only=True, use_safetensors=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation="sdpa",
    ).eval().to("cuda")
    observed = {key: int(getattr(vjepa.config, key)) for key in
                ("frames_per_clip", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_labels")}
    expected = {"frames_per_clip": 16, "hidden_size": 1024, "num_hidden_layers": 24,
                "num_attention_heads": 16, "num_labels": 174}
    if observed != expected:
        raise ExperimentError(f"V-JEPA2 config drift: {observed}")
    sig_processor = AutoProcessor.from_pretrained(SIGLIP_REPO, revision=SIGLIP_REVISION, trust_remote_code=False)
    siglip = AutoModel.from_pretrained(
        SIGLIP_REPO, revision=SIGLIP_REVISION, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float16,
    ).eval().to("cuda")
    resolved = getattr(siglip.config, "_commit_hash", None)
    if resolved != SIGLIP_REVISION:
        raise ExperimentError(f"SigLIP revision drift: {resolved}")
    for model in (vjepa, siglip):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return vjepa, siglip, sig_processor, {"vjepa_files": model_facts, "vjepa_config": observed, "siglip_revision": resolved}


def siglip_text_features(model: Any, processor: Any, values: list[str], torch: Any) -> np.ndarray:
    prompts = [f"a video of a person acting {value.lower()}" for value in values]
    batch = processor(text=prompts, padding="max_length", return_tensors="pt")
    batch = {key: value.to("cuda") for key, value in batch.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model.get_text_features(**batch)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    result = features.cpu().numpy().astype(np.float32)
    if result.shape != (len(values), 768) or not np.isfinite(result).all():
        raise ExperimentError("SigLIP text feature contract failed")
    return result


def extract_path_features(
    depth_frames: np.ndarray, ir_frames: np.ndarray, depth_pace: np.ndarray, ir_pace: np.ndarray,
    vjepa: Any, siglip: Any, sig_processor: Any, torch: Any,
) -> np.ndarray:
    import torch.nn.functional as functional
    vj_blocks = []
    with torch.inference_mode():
        for t in range(WINDOWS):
            raw = np.stack((depth_frames[t], ir_frames[t]))
            batch = torch.from_numpy(raw).permute(0, 1, 4, 2, 3).float().div_(255.0)
            flat = batch.reshape(2 * FRAMES, 3, SOURCE_SIZE, SOURCE_SIZE)
            flat = functional.interpolate(flat, size=(VJEPA_RESIZE, VJEPA_RESIZE), mode="bilinear", align_corners=False)
            offset = (VJEPA_RESIZE - VJEPA_SIZE) // 2
            batch = flat[:, :, offset:offset + VJEPA_SIZE, offset:offset + VJEPA_SIZE].reshape(2, FRAMES, 3, VJEPA_SIZE, VJEPA_SIZE)
            mean = torch.from_numpy(MEAN).view(1, 1, 3, 1, 1)
            std = torch.from_numpy(STD).view(1, 1, 3, 1, 1)
            batch = batch.sub(mean).div(std).to("cuda", dtype=torch.float16)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                encoded = vjepa.vjepa2(pixel_values_videos=batch, skip_predictor=True, return_dict=True)
                pooled = vjepa.pooler(encoded.last_hidden_state)
                logits = vjepa.classifier(pooled)
            pooled = functional.normalize(pooled.float(), dim=-1).cpu().numpy()
            logits = logits.float().cpu().numpy()
            block = np.concatenate((pooled[0], pooled[1], np.abs(pooled[0] - pooled[1]),
                                    logits[0], logits[1], np.abs(logits[0] - logits[1])), axis=0)
            vj_blocks.append(block.astype(np.float32))
    from PIL import Image
    center_ir = [Image.fromarray(ir_frames[t, FRAMES // 2]) for t in range(WINDOWS)]
    sig_batch = sig_processor(images=center_ir, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        sig_features = siglip.get_image_features(pixel_values=sig_batch)
    if not isinstance(sig_features, torch.Tensor):
        sig_features = sig_features.pooler_output
    sig_features = functional.normalize(sig_features.float(), dim=-1).cpu().numpy()
    result = np.concatenate((np.stack(vj_blocks), sig_features, depth_pace, ir_pace), axis=1).astype(np.float32)
    if result.shape != (WINDOWS, 4374) or not np.isfinite(result).all():
        raise ExperimentError(f"joint feature shape drift: {result.shape}")
    return result


def extract_all_features(
    archive_path: Path, paths: list[str], targets: dict[str, Any], output_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    if not torch.cuda.is_available():
        raise ExperimentError("CUDA GPU is mandatory")
    vjepa, siglip, sig_processor, model_receipt = acquire_models(Path("/kaggle/temp/cuhkx_joint_models"))
    action_text = siglip_text_features(siglip, sig_processor, list(ACTIONS), torch)
    emotion_text = siglip_text_features(siglip, sig_processor, targets["emotion_terms"], torch)
    features = np.empty((len(paths), WINDOWS, 4374), dtype=np.float16)
    temp_depth = output_root / "train_depth_color.mp4"
    temp_ir = output_root / "train_ir.mp4"
    started = time.time()
    with zipfile.ZipFile(archive_path) as archive:
        depth_members, ir_members = build_member_maps(archive, paths)
        for i, path in enumerate(paths):
            with archive.open(depth_members[path]) as source, temp_depth.open("wb") as target:
                shutil.copyfileobj(source, target)
            with archive.open(ir_members[path]) as source, temp_ir.open("wb") as target:
                shutil.copyfileobj(source, target)
            depth_frames, depth_pace = decode_windows(temp_depth, False)
            ir_frames, ir_pace = decode_windows(temp_ir, True)
            features[i] = extract_path_features(
                depth_frames, ir_frames, depth_pace, ir_pace, vjepa, siglip, sig_processor, torch
            ).astype(np.float16)
            if (i + 1) % 10 == 0 or i + 1 == len(paths):
                print(json.dumps({"feature_paths": i + 1, "total": len(paths),
                                  "elapsed_minutes": round((time.time() - started) / 60, 2)}), flush=True)
    temp_depth.unlink(missing_ok=True)
    temp_ir.unlink(missing_ok=True)
    if not np.isfinite(features.astype(np.float32)).all() or float(features.std()) <= 0:
        raise ExperimentError("joint features are non-finite or constant")
    del vjepa, siglip
    torch.cuda.empty_cache()
    return features, action_text, emotion_text, {
        "elapsed_seconds": time.time() - started,
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "model": model_receipt,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
    }


def make_head(input_dim: int, action_text: Any, emotion_text: Any, torch: Any):
    nn = torch.nn

    class TimelineHead(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = 192
            self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.15))
            self.position = nn.Parameter(torch.zeros(1, WINDOWS, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=384, dropout=.15,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.action_semantic = nn.Linear(hidden, 768)
            self.action_free = nn.Linear(hidden, 40)
            self.emotion_semantic = nn.Linear(hidden * 3, 768)
            self.action_scale = nn.Parameter(torch.tensor(math.log(12.0)))
            self.emotion_scale = nn.Parameter(torch.tensor(math.log(12.0)))
            self.register_buffer("action_text", action_text)
            self.register_buffer("emotion_text", emotion_text)
            nn.init.normal_(self.position, std=.01)

        def forward(self, value):
            import torch.nn.functional as f
            hidden = self.temporal(self.input(value) + self.position)
            semantic = f.normalize(self.action_semantic(hidden), dim=-1)
            window = self.action_free(hidden) + self.action_scale.clamp(0, math.log(100)).exp() * (semantic @ self.action_text.T)
            tau = .50
            presence = tau * torch.logsumexp(window / tau, dim=1) - tau * math.log(WINDOWS)
            presence = .75 * presence + .25 * window.mean(dim=1)
            positions = torch.linspace(0.0, 1.0, WINDOWS, device=value.device).view(1, WINDOWS, 1)
            occurrence = (torch.softmax(window / .35, dim=1) * positions).sum(dim=1)
            pooled = torch.cat((hidden.mean(1), hidden.std(1, unbiased=False), hidden.amax(1)), dim=1)
            emotion_query = f.normalize(self.emotion_semantic(pooled), dim=-1)
            emotion = self.emotion_scale.clamp(0, math.log(100)).exp() * (emotion_query @ self.emotion_text.T)
            return presence, occurrence, emotion

    return TimelineHead()


def fold_loss(presence: Any, occurrence: Any, emotion: Any, target: dict[str, Any], train: Any, torch: Any):
    import torch.nn.functional as functional
    known = target["known"][train]
    truth = target["presence"][train]
    positives = (truth * known).sum(0)
    negatives = ((1.0 - truth) * known).sum(0)
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    raw = functional.binary_cross_entropy_with_logits(presence, truth, pos_weight=pos_weight, reduction="none")
    presence_loss = (raw * known).sum() / known.sum().clamp_min(1)

    single_scores = presence.gather(1, target["single_opts"][train])
    single_loss = functional.cross_entropy(single_scores, target["single_ans"][train])
    multi_scores = presence.gather(1, target["multi_opts"][train])
    multi_loss = functional.binary_cross_entropy_with_logits(multi_scores, target["multi_targets"][train])

    combo_valid = target["combo_ans"][train].ge(0)
    combo_masks = target["combo_opts"][train][combo_valid]
    combo_presence = presence[combo_valid]
    union = combo_masks.amax(dim=1, keepdim=True)
    positive = (combo_masks * functional.logsigmoid(combo_presence[:, None, :])).sum(2) / combo_masks.sum(2).clamp_min(1)
    negative_mask = (union - combo_masks).clamp_min(0)
    negative = (negative_mask * functional.logsigmoid(-combo_presence[:, None, :])).sum(2) / negative_mask.sum(2).clamp_min(1)
    combo_scores = positive + .50 * negative
    combo_loss = functional.cross_entropy(combo_scores, target["combo_ans"][train][combo_valid])

    seq_global = target["sequence_paths"]
    seq_mask_np = np.isin(seq_global.detach().cpu().numpy(), train.detach().cpu().numpy())
    seq_mask = torch.from_numpy(seq_mask_np).to(train.device)
    seq_paths = seq_global[seq_mask]
    path_remap = torch.full((EXPECTED_PATHS,), -1, dtype=torch.long, device=train.device)
    path_remap[train] = torch.arange(len(train), device=train.device)
    local_paths = path_remap[seq_paths]
    orders = target["sequence_order"][seq_mask]
    seq_times = occurrence[local_paths].gather(1, orders)
    pair_terms = [functional.softplus((seq_times[:, a] - seq_times[:, b]) * 8.0).mean()
                  for a in range(4) for b in range(a + 1, 4)]
    sequence_loss = torch.stack(pair_terms).mean()

    emotion_scores = emotion.gather(1, target["emotion_opts"][train])
    emotion_loss = functional.cross_entropy(emotion_scores, target["emotion_ans"][train])
    total = presence_loss + .50 * single_loss + multi_loss + combo_loss + sequence_loss + emotion_loss
    return total, {"presence": presence_loss, "single": single_loss, "multi": multi_loss,
                   "combination": combo_loss, "sequence": sequence_loss, "emotion": emotion_loss}


def tensor_targets(target: dict[str, Any], device: Any, torch: Any) -> dict[str, Any]:
    out = {}
    for key, value in target.items():
        if isinstance(value, np.ndarray):
            if value.dtype == bool:
                out[key] = torch.from_numpy(value).to(device=device, dtype=torch.float32)
            elif np.issubdtype(value.dtype, np.floating):
                out[key] = torch.from_numpy(value).to(device=device, dtype=torch.float32)
            else:
                out[key] = torch.from_numpy(value).to(device=device, dtype=torch.long)
    return out


def train_sixfold(
    features: np.ndarray, users: np.ndarray, targets: dict[str, Any], action_text: np.ndarray, emotion_text: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    import torch
    device = torch.device("cuda")
    raw = torch.from_numpy(features.astype(np.float32)).to(device)
    tensor_target = tensor_targets(targets, device, torch)
    action_text_tensor = torch.from_numpy(action_text).to(device)
    emotion_text_tensor = torch.from_numpy(emotion_text).to(device)
    oof_presence = np.empty((EXPECTED_PATHS, 40), dtype=np.float32)
    oof_occurrence = np.empty((EXPECTED_PATHS, 40), dtype=np.float32)
    oof_emotion = np.empty((EXPECTED_PATHS, len(targets["emotion_terms"])), dtype=np.float32)
    summaries = []
    all_indices = np.arange(EXPECTED_PATHS)
    for fold, held_users in enumerate(FOLD_USERS):
        random.seed(SEED + fold)
        np.random.seed(SEED + fold)
        torch.manual_seed(SEED + fold)
        torch.cuda.manual_seed_all(SEED + fold)
        held_np = np.flatnonzero(np.isin(users, held_users))
        train_np = np.setdiff1d(all_indices, held_np, assume_unique=True)
        train = torch.from_numpy(train_np).to(device=device, dtype=torch.long)
        held = torch.from_numpy(held_np).to(device=device, dtype=torch.long)
        mean = raw[train].mean((0, 1), keepdim=True)
        std = raw[train].std((0, 1), unbiased=False, keepdim=True).clamp_min(.02)
        x_train = ((raw[train] - mean) / std).clamp(-8, 8)
        x_held = ((raw[held] - mean) / std).clamp(-8, 8)
        head = make_head(raw.shape[-1], action_text_tensor, emotion_text_tensor, torch).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=220, eta_min=2e-5)
        final_parts = {}
        started = time.time()
        for epoch in range(220):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            presence, occurrence, emotion = head(x_train)
            loss, parts = fold_loss(presence, occurrence, emotion, tensor_target, train, torch)
            if not torch.isfinite(loss):
                raise ExperimentError(f"non-finite fold {fold} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            final_parts = {key: float(value.detach().cpu()) for key, value in parts.items()}
            if (epoch + 1) % 55 == 0:
                print(json.dumps({"fold": fold, "epoch": epoch + 1, "loss": float(loss.detach().cpu()),
                                  "parts": final_parts}), flush=True)
        head.eval()
        with torch.inference_mode():
            presence, occurrence, emotion = head(x_held)
        oof_presence[held_np] = presence.cpu().numpy()
        oof_occurrence[held_np] = occurrence.cpu().numpy()
        oof_emotion[held_np] = emotion.cpu().numpy()
        summaries.append({"fold": fold, "held_users": list(held_users), "train_paths": len(train_np),
                          "held_paths": len(held_np), "seconds": time.time() - started,
                          "final_training_losses": final_parts})
        del head, optimizer, scheduler, x_train, x_held
        torch.cuda.empty_cache()
    if not np.isfinite(oof_presence).all() or not np.isfinite(oof_occurrence).all() or not np.isfinite(oof_emotion).all():
        raise ExperimentError("non-finite OOF output")
    return oof_presence, oof_occurrence, oof_emotion, summaries


def stable_softmax(value: np.ndarray) -> np.ndarray:
    shifted = value.astype(np.float64) - value.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def decode_oof(
    rows: list[dict[str, str]], paths: list[str], targets: dict[str, Any], users: np.ndarray,
    presence: np.ndarray, occurrence: np.ndarray, emotion: np.ndarray,
) -> list[dict[str, Any]]:
    pindex = {path: i for i, path in enumerate(paths)}
    output = []
    for row in rows:
        i = pindex[row["path"]]
        cat = row["category"]
        if cat == "single":
            scores = presence[i, targets["single_opts"][i]]
            prob = stable_softmax(scores[None])[0]
            order = np.argsort(-prob)
            pred = LETTERS[int(order[0])]
            confidence, margin = float(prob[order[0]]), float(prob[order[0]] - prob[order[1]])
        elif cat == "multi":
            scores = presence[i, targets["multi_opts"][i]]
            prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
            chosen = [letter for letter, value in zip(LETTERS, prob) if value >= .5]
            if not chosen:
                chosen = [LETTERS[int(np.argmax(prob))]]
            pred = "".join(chosen)
            confidence = float(np.mean(np.maximum(prob, 1.0 - prob)))
            margin = float(np.min(np.abs(prob - .5)) * 2.0)
        elif cat == "combination":
            masks = targets["combo_opts"][i]
            union = masks.max(axis=0, keepdims=True)
            logp = -np.logaddexp(0, -presence[i])
            logn = -np.logaddexp(0, presence[i])
            positive = (masks * logp[None]).sum(1) / np.maximum(masks.sum(1), 1)
            negative_mask = np.maximum(union - masks, 0)
            negative = (negative_mask * logn[None]).sum(1) / np.maximum(negative_mask.sum(1), 1)
            prob = stable_softmax((positive + .5 * negative)[None])[0]
            order = np.argsort(-prob)
            pred = LETTERS[int(order[0])]
            confidence, margin = float(prob[order[0]]), float(prob[order[0]] - prob[order[1]])
        elif cat == "sequence":
            actions = [parse_actions(row[letter])[0] for letter in LETTERS]
            times = occurrence[i, actions]
            order = np.argsort(times, kind="stable")
            pred = "".join(LETTERS[int(j)] for j in order)
            gaps = np.diff(np.sort(times))
            confidence = float(np.clip(gaps.mean() * 4.0 if len(gaps) else 0.0, 0, 1))
            margin = float(np.clip(gaps.min() * 4.0 if len(gaps) else 0.0, 0, 1))
        elif cat == "emotion":
            scores = emotion[i, targets["emotion_opts"][i]]
            prob = stable_softmax(scores[None])[0]
            order = np.argsort(-prob)
            pred = LETTERS[int(order[0])]
            confidence, margin = float(prob[order[0]]), float(prob[order[0]] - prob[order[1]])
        else:
            raise ExperimentError(cat)
        fold = next(index for index, held in enumerate(FOLD_USERS) if int(users[i]) in held)
        output.append({"qa_id": row["qa_id"], "path": row["path"], "user": int(users[i]), "fold": fold,
                       "category": cat, "answer": row["answer"], "prediction": pred,
                       "confidence": confidence, "margin": margin})
    return output


def metrics(rows: list[dict[str, Any]], targets: dict[str, Any], presence: np.ndarray, occurrence: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    result = {}
    for category in ("single", "multi", "combination", "sequence", "emotion"):
        selected = [row for row in rows if row["category"] == category]
        result[category] = {"rows": len(selected), "correct": sum(row["prediction"] == row["answer"] for row in selected),
                            "accuracy": float(np.mean([row["prediction"] == row["answer"] for row in selected]))}
    result["overall"] = {"rows": len(rows), "correct": sum(row["prediction"] == row["answer"] for row in rows),
                         "accuracy": float(np.mean([row["prediction"] == row["answer"] for row in rows]))}
    result["fold_action_qa_accuracy"] = []
    for fold in range(6):
        selected = [row for row in rows if row["fold"] == fold and row["category"] != "emotion"]
        result["fold_action_qa_accuracy"].append(float(np.mean([row["prediction"] == row["answer"] for row in selected])))
    exact = targets["combo_ans"] >= 0
    exact_truth = targets["presence"][exact]
    exact_prob = 1.0 / (1.0 + np.exp(-np.clip(presence[exact], -30, 30)))
    aucs = []
    for action in range(40):
        if len(np.unique(exact_truth[:, action])) == 2:
            aucs.append(float(roc_auc_score(exact_truth[:, action], exact_prob[:, action])))
    pred_set = exact_prob >= .5
    intersection = (pred_set & exact_truth.astype(bool)).sum(1)
    union = (pred_set | exact_truth.astype(bool)).sum(1)
    result["action_representation"] = {
        "atomic_macro_auc": float(np.mean(aucs)), "atomic_actions_scored": len(aucs),
        "exact_set_jaccard": float(np.mean(intersection / np.maximum(union, 1))),
        "exact_set_accuracy": float(np.mean(np.all(pred_set == exact_truth.astype(bool), axis=1))),
    }
    orders = targets["sequence_order"]
    seq_paths = targets["sequence_paths"]
    ordered_times = occurrence[seq_paths[:, None], orders]
    pair_correct = [(ordered_times[:, a] < ordered_times[:, b]).astype(np.float32)
                    for a in range(4) for b in range(a + 1, 4)]
    result["sequence_pairwise_accuracy"] = float(np.mean(np.stack(pair_correct)))
    gate = PROTOCOL["screen_gate"]
    result["screen_pass"] = bool(
        result["action_representation"]["atomic_macro_auc"] >= gate["atomic_macro_auc_min"]
        and result["action_representation"]["exact_set_jaccard"] >= gate["action_set_jaccard_min"]
        and result["combination"]["accuracy"] >= gate["combination_four_option_accuracy_min"]
        and result["sequence_pairwise_accuracy"] >= gate["sequence_pairwise_accuracy_min"]
        and min(result["fold_action_qa_accuracy"]) >= gate["all_fold_action_qa_accuracy_min"]
    )
    return result


def anonymous_unit(value: str) -> str:
    match = re.search(r"(?:^|/)(large_model_track_test/LM_test_\d+)(?:/|$)", value.replace("\\", "/"))
    if match is None:
        raise ExperimentError(f"invalid anonymous path: {value}")
    return match.group(1)


def load_test_qa(path: Path) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = [dict(row) for row in csv.DictReader(handle)]
    if any(str(row.get("answer", "")).strip() or str(row.get("prediction", "")).strip() for row in all_rows):
        raise ExperimentError("refusing anonymous table with populated answer or prediction")
    rows = [row for row in all_rows if row.get("source") == "HAU"]
    if len(rows) != EXPECTED_TEST_QA or len({row["qa_id"] for row in rows}) != EXPECTED_TEST_QA:
        raise ExperimentError(f"anonymous HAU QA count drift: {len(rows)}")
    for row in rows:
        row["unit"] = anonymous_unit(row["path"])
        if row["category"] not in {"single", "multi", "combination", "sequence", "emotion"}:
            raise ExperimentError(f"unexpected anonymous HAU category: {row['category']}")
        if any(not str(row.get(letter, "")).strip() for letter in LETTERS):
            raise ExperimentError(f"empty anonymous option: {row['qa_id']}")
    paths = sorted({row["unit"] for row in rows})
    if len(paths) != EXPECTED_TEST_PATHS:
        raise ExperimentError(f"anonymous HAU path count drift: {len(paths)}")
    categories = {name: sum(row["category"] == name for row in rows)
                  for name in ("single", "multi", "combination", "sequence", "emotion")}
    expected = {"single": 144, "multi": 144, "combination": 139, "sequence": 39, "emotion": 144}
    if categories != expected:
        raise ExperimentError(f"anonymous category drift: {categories}")
    return rows, paths, {"rows": len(rows), "paths": len(paths), "category_counts": categories,
                         "paths_sha256": canonical_sha(paths)}


def find_test_media_root(input_root: Path) -> Path:
    sentinels = [path for path in input_root.rglob("LM_test_0001/IR/IR.mp4")
                 if "cuhk-x-qwen2-5-vl-large-model-track-media-relay" in str(path).replace("\\", "/")]
    roots = sorted({path.parents[2].resolve() for path in sentinels})
    if len(roots) != 1:
        raise ExperimentError(f"expected one exact anonymous media relay root, got {roots}")
    return roots[0]


def extract_train_test_features(
    archive_path: Path, train_paths: list[str], test_root: Path, test_paths: list[str],
    targets: dict[str, Any], output_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    if not torch.cuda.is_available():
        raise ExperimentError("CUDA GPU is mandatory")
    vjepa, siglip, sig_processor, model_receipt = acquire_models(Path("/kaggle/temp/cuhkx_joint_models"))
    action_text = siglip_text_features(siglip, sig_processor, list(ACTIONS), torch)
    emotion_text = siglip_text_features(siglip, sig_processor, targets["emotion_terms"], torch)
    train_features = np.empty((len(train_paths), WINDOWS, 4374), dtype=np.float16)
    test_features = np.empty((len(test_paths), WINDOWS, 4374), dtype=np.float16)
    temp_depth = output_root / "current_depth_color.mp4"
    temp_ir = output_root / "current_ir.mp4"
    started = time.time()
    with zipfile.ZipFile(archive_path) as archive:
        depth_members, ir_members = build_member_maps(archive, train_paths)
        for i, path in enumerate(train_paths):
            with archive.open(depth_members[path]) as source, temp_depth.open("wb") as target:
                shutil.copyfileobj(source, target)
            with archive.open(ir_members[path]) as source, temp_ir.open("wb") as target:
                shutil.copyfileobj(source, target)
            depth_frames, depth_pace = decode_windows(temp_depth, False)
            ir_frames, ir_pace = decode_windows(temp_ir, True)
            train_features[i] = extract_path_features(
                depth_frames, ir_frames, depth_pace, ir_pace, vjepa, siglip, sig_processor, torch
            ).astype(np.float16)
            if (i + 1) % 10 == 0 or i + 1 == len(train_paths):
                print(json.dumps({"phase": "train_features", "paths": i + 1, "total": len(train_paths),
                                  "elapsed_minutes": round((time.time() - started) / 60, 2)}), flush=True)
    depth_proxy_units = []
    for i, path in enumerate(test_paths):
        clip = path.rsplit("/", 1)[-1]
        depth_path = test_root / clip / "Depth_Color" / "Depth_Color.mp4"
        ir_path = test_root / clip / "IR" / "IR.mp4"
        if not ir_path.is_file():
            raise ExperimentError(f"anonymous media coverage missing: {path}")
        if not depth_path.is_file():
            if path not in EXPECTED_TEST_DEPTH_PROXY_UNITS:
                raise ExperimentError(f"unexpected anonymous depth coverage missing: {path}")
            depth_path = ir_path
            depth_proxy_units.append(path)
        depth_frames, depth_pace = decode_windows(depth_path, False)
        ir_frames, ir_pace = decode_windows(ir_path, True)
        test_features[i] = extract_path_features(
            depth_frames, ir_frames, depth_pace, ir_pace, vjepa, siglip, sig_processor, torch
        ).astype(np.float16)
        if (i + 1) % 10 == 0 or i + 1 == len(test_paths):
            print(json.dumps({"phase": "test_features", "paths": i + 1, "total": len(test_paths),
                              "elapsed_minutes": round((time.time() - started) / 60, 2)}), flush=True)
    if tuple(depth_proxy_units) != EXPECTED_TEST_DEPTH_PROXY_UNITS:
        raise ExperimentError(f"anonymous depth proxy census drift: {depth_proxy_units}")
    temp_depth.unlink(missing_ok=True)
    temp_ir.unlink(missing_ok=True)
    if not np.isfinite(train_features.astype(np.float32)).all() or not np.isfinite(test_features.astype(np.float32)).all():
        raise ExperimentError("non-finite fullfit features")
    del vjepa, siglip
    torch.cuda.empty_cache()
    return train_features, test_features, action_text, emotion_text, {
        "elapsed_seconds": time.time() - started,
        "train_shape": list(train_features.shape), "test_shape": list(test_features.shape),
        "dtype": str(train_features.dtype), "model": model_receipt,
        "anonymous_depth_proxy_units": depth_proxy_units,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
    }


def train_fullfit_ensemble(
    train_features: np.ndarray, test_features: np.ndarray, targets: dict[str, Any],
    action_text: np.ndarray, emotion_text: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    import torch
    device = torch.device("cuda")
    raw_train = torch.from_numpy(train_features.astype(np.float32)).to(device)
    raw_test = torch.from_numpy(test_features.astype(np.float32)).to(device)
    mean = raw_train.mean((0, 1), keepdim=True)
    std = raw_train.std((0, 1), unbiased=False, keepdim=True).clamp_min(.02)
    x_train = ((raw_train - mean) / std).clamp(-8, 8)
    x_test = ((raw_test - mean) / std).clamp(-8, 8)
    tensor_target = tensor_targets(targets, device, torch)
    train = torch.arange(len(train_features), device=device, dtype=torch.long)
    action_text_tensor = torch.from_numpy(action_text).to(device)
    emotion_text_tensor = torch.from_numpy(emotion_text).to(device)
    presence_parts, occurrence_parts, emotion_parts, summaries = [], [], [], []
    for member in range(6):
        seed = SEED + 100 + member
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        head = make_head(raw_train.shape[-1], action_text_tensor, emotion_text_tensor, torch).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=220, eta_min=2e-5)
        final_parts: dict[str, float] = {}
        started = time.time()
        for epoch in range(220):
            head.train(); optimizer.zero_grad(set_to_none=True)
            presence, occurrence, emotion = head(x_train)
            loss, parts = fold_loss(presence, occurrence, emotion, tensor_target, train, torch)
            if not torch.isfinite(loss):
                raise ExperimentError(f"non-finite fullfit member {member} loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            final_parts = {key: float(value.detach().cpu()) for key, value in parts.items()}
            if (epoch + 1) % 55 == 0:
                print(json.dumps({"phase": "fullfit", "member": member, "epoch": epoch + 1,
                                  "loss": float(loss.detach().cpu())}), flush=True)
        head.eval()
        export_dir = Path(OOF_MEMBER_EXPORT_DIR)
        export_dir.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()},
                   export_dir / f"member-{member}.pt")
        if member == 0:
            np.savez_compressed(export_dir / "normalization.npz", mean=mean.detach().cpu().numpy(), std=std.detach().cpu().numpy())
            np.savez_compressed(export_dir / "text_embeddings.npz", action_text=action_text,
                                emotion_text=emotion_text, emotion_terms=np.asarray(targets["emotion_terms"],dtype="U64"))
        with torch.inference_mode():
            presence, occurrence, emotion = head(x_test)
        presence_parts.append(presence.cpu().numpy())
        occurrence_parts.append(occurrence.cpu().numpy())
        emotion_parts.append(emotion.cpu().numpy())
        summaries.append({"member": member, "seed": seed, "seconds": time.time() - started,
                          "final_training_losses": final_parts})
        del head, optimizer, scheduler
        torch.cuda.empty_cache()
    return (np.mean(presence_parts, axis=0).astype(np.float32),
            np.mean(occurrence_parts, axis=0).astype(np.float32),
            np.mean(emotion_parts, axis=0).astype(np.float32), summaries)


def decode_test(
    rows: list[dict[str, str]], paths: list[str], emotion_terms: list[str],
    presence: np.ndarray, occurrence: np.ndarray, emotion: np.ndarray,
) -> list[dict[str, Any]]:
    pindex = {path: i for i, path in enumerate(paths)}
    eindex = {value: i for i, value in enumerate(emotion_terms)}
    output = []
    for row in rows:
        i, cat = pindex[row["unit"]], row["category"]
        if cat == "single":
            actions = [parse_actions(row[letter])[0] for letter in LETTERS]
            prob = stable_softmax(presence[i, actions][None])[0]
            order = np.argsort(-prob); pred = LETTERS[int(order[0])]
            confidence, margin = float(prob[order[0]]), float(prob[order[0]] - prob[order[1]])
        elif cat == "multi":
            actions = [parse_actions(row[letter])[0] for letter in LETTERS]
            prob = 1.0 / (1.0 + np.exp(-np.clip(presence[i, actions], -30, 30)))
            chosen = [letter for letter, value in zip(LETTERS, prob) if value >= .5]
            if not chosen: chosen = [LETTERS[int(np.argmax(prob))]]
            pred = "".join(chosen); confidence = float(np.mean(np.maximum(prob, 1.0 - prob)))
            margin = float(np.min(np.abs(prob - .5)) * 2.0)
        elif cat == "combination":
            masks = np.zeros((4, 40), dtype=np.float32)
            for j, letter in enumerate(LETTERS): masks[j, parse_actions(row[letter])] = 1.0
            union = masks.max(axis=0, keepdims=True)
            logp = -np.logaddexp(0, -presence[i]); logn = -np.logaddexp(0, presence[i])
            positive = (masks * logp[None]).sum(1) / np.maximum(masks.sum(1), 1)
            negative_mask = np.maximum(union - masks, 0)
            negative = (negative_mask * logn[None]).sum(1) / np.maximum(negative_mask.sum(1), 1)
            prob = stable_softmax((positive + .5 * negative)[None])[0]
            order = np.argsort(-prob); pred = LETTERS[int(order[0])]
            confidence, margin = float(prob[order[0]]), float(prob[order[0]] - prob[order[1]])
        elif cat == "sequence":
            actions = [parse_actions(row[letter])[0] for letter in LETTERS]
            times = occurrence[i, actions]; order = np.argsort(times, kind="stable")
            pred = "".join(LETTERS[int(j)] for j in order); gaps = np.diff(np.sort(times))
            confidence = float(np.clip(gaps.mean() * 4.0, 0, 1)); margin = float(np.clip(gaps.min() * 4.0, 0, 1))
        elif cat == "emotion":
            try: options = [eindex[row[letter]] for letter in LETTERS]
            except KeyError as error: raise ExperimentError(f"unknown anonymous emotion term: {error}") from error
            prob = stable_softmax(emotion[i, options][None])[0]
            order = np.argsort(-prob); pred = LETTERS[int(order[0])]
            confidence, margin = float(prob[order[0]]), float(prob[order[0]] - prob[order[1]])
        else:
            raise ExperimentError(cat)
        output.append({"qa_id": row["qa_id"], "unit": row["unit"], "category": cat,
                       "prediction": pred, "confidence": confidence, "margin": margin})
    if len(output) != EXPECTED_TEST_QA or len({row["qa_id"] for row in output}) != EXPECTED_TEST_QA:
        raise ExperimentError("anonymous output contract failed")
    return output


def main() -> None:
    output_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    source_path = Path(__file__).resolve()
    protocol_sha = canonical_sha(PROTOCOL)
    expected_auth = {"test_model_inference": True, "candidate": False, "submission": False}
    if PROTOCOL["authorization"] != expected_auth:
        raise ExperimentError("authorization drift")
    if tuple(sorted(ACTIONS)) != ACTIONS or len(ACTIONS) != 40 or len(set(ACTIONS)) != 40:
        raise ExperimentError("action ontology drift")
    if "--static-check" in sys.argv[1:]:
        print(json.dumps({"status": "STATIC_PASS", "protocol_sha256": protocol_sha,
                          "source_sha256": sha256(source_path), "candidate": False, "submission": False}))
        return
    started = time.time()
    runtime = install_runtime()
    input_root = Path("/kaggle/input")
    train_qa = find_exact(input_root, "training_qa.csv", expected_sha=TRAIN_QA_SHA256)
    test_qa = find_exact(input_root, "test_qa.csv", expected_sha=TEST_QA_SHA256)
    archive = find_exact(input_root, "HAU.zip", expected_bytes=HAU_ARCHIVE_BYTES)
    test_root = find_test_media_root(input_root)
    train_rows, train_paths, users, train_receipt = load_qa(train_qa)
    test_rows, test_paths, test_receipt = load_test_qa(test_qa)
    targets = build_targets(train_rows, train_paths)

    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 0) and "sm_60" not in torch.cuda.get_arch_list():
        raise ExperimentError(f"P100 runtime incompatibility: torch={torch.__version__}, arch={torch.cuda.get_arch_list()}")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    train_features, test_features, action_text, emotion_text, extraction = extract_train_test_features(
        archive, train_paths, test_root, test_paths, targets, output_root
    )
    presence, occurrence, emotion, members = train_fullfit_ensemble(
        train_features, test_features, targets, action_text, emotion_text
    )
    decoded = decode_test(test_rows, test_paths, targets["emotion_terms"], presence, occurrence, emotion)
    prediction_path = output_root / "vjepa_siglip_hau_fullfit_test_predictions_v1.csv"
    with prediction_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decoded[0])); writer.writeheader(); writer.writerows(decoded)
    feature_path = output_root / "vjepa_siglip_hau_fullfit_features_v1.npz"
    with feature_path.open("xb") as handle:
        np.savez_compressed(handle, train_paths=np.asarray(train_paths, dtype="U32"),
                            test_paths=np.asarray(test_paths, dtype="U48"),
                            train_features=train_features, test_features=test_features,
                            action_text=action_text.astype(np.float16), emotion_text=emotion_text.astype(np.float16),
                            emotion_terms=np.asarray(targets["emotion_terms"], dtype="U64"),
                            presence_logits=presence.astype(np.float16), occurrence_times=occurrence.astype(np.float16),
                            emotion_logits=emotion.astype(np.float16))
    report = {
        "status": "COMPLETE_MODEL_PREDICTIONS_ONLY", "schema": "CUHKX_HAU_VJEPA_SIGLIP_FULLFIT_INFERENCE_V1",
        "training_only_gate": PROTOCOL["train_only_gate"], "anonymous_media_accessed": True,
        "anonymous_answers_available_or_accessed": False, "candidate": False, "submission": False,
        "protocol": PROTOCOL, "protocol_sha256": protocol_sha,
        "source": {"bytes": source_path.stat().st_size, "sha256": sha256(source_path)},
        "runtime_versions": runtime,
        "inputs": {"training_qa": {"bytes": train_qa.stat().st_size, "sha256": sha256(train_qa)},
                   "test_qa": {"bytes": test_qa.stat().st_size, "sha256": sha256(test_qa)},
                   "hau_archive": {"bytes": archive.stat().st_size}, "test_media_root": str(test_root),
                   "train_receipt": train_receipt, "test_receipt": test_receipt},
        "extraction": extraction, "ensemble_members": members,
        "outputs": {prediction_path.name: {"bytes": prediction_path.stat().st_size, "sha256": sha256(prediction_path)},
                    feature_path.name: {"bytes": feature_path.stat().st_size, "sha256": sha256(feature_path)}},
        "elapsed_seconds": time.time() - started,
        "firewall": {"test_labels": False, "manual_test_media": False, "aim_histograms": False,
                     "singleton_probes": False, "row_level_leaderboard_inference": False,
                     "candidate_or_submission_written": False},
    }
    report_path = output_root / "VJEPA_SIGLIP_HAU_FULLFIT_INFERENCE_V1.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": report["status"], "predictions": len(decoded),
                      "report_sha256": sha256(report_path), "elapsed_seconds": report["elapsed_seconds"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
