from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SCHEMA = "cuhkx/smalltrack/dst-ridge-rowlocal-v2"
CLASSES = 40
FRAME_PROJECTED_DIMENSION = 256
FEATURE_DIMENSIONS = (1024, 2048)
SEED = 1729
FRAME_SEED = 1730
VIEWS = (1.75, 2.0, 2.25)
DEFAULT_FRAME_BATCH_SIZE = 512
RIDGE_ALPHA = 1.0e-8
EXPECTED_DST_STATE_ELEMENTS = 2_460_462
EXPECTED_DST_STATE_SHA256 = "eab5a0735982e39470ad81a16bf8bce1a0b37f98ace5d4690e97082d7678e53d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_contract(feature_dimension: int) -> dict[str, Any]:
    if feature_dimension not in FEATURE_DIMENSIONS:
        raise RuntimeError("feature dimension is outside the prelocked escalation")
    return {
        "schema": SCHEMA,
        "classes": CLASSES,
        "feature_dimension": feature_dimension,
        "frame_projected_dimension": FRAME_PROJECTED_DIMENSION,
        "feature_seed": SEED,
        "frame_seed": FRAME_SEED,
        "views": VIEWS,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_solver": "svd",
        "head": "standardize_then_affine",
        "sequence_statistics": "mean_std_max_bins2_bins3_absdiff_mean_std_direct_probability_log_count",
        "row_local": True,
        "post_head_decoder": False,
    }


def flatten_valid(data: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int32)
    if data.ndim != 4 or data.shape[1:] != (128, 17, 3) or counts.shape != (len(data),) or not len(data):
        raise RuntimeError("skeleton sequence closure mismatch")
    frames: list[np.ndarray] = []
    offsets = [0]
    for row, count_raw in enumerate(counts.tolist()):
        count = int(count_raw)
        if count <= 0 or count > data.shape[1]:
            raise RuntimeError(f"invalid frame count at row {row}: {count}")
        frames.append(np.ascontiguousarray(data[row, :count], dtype=np.float32))
        offsets.append(offsets[-1] + count)
    return np.concatenate(frames), np.asarray(offsets, dtype=np.int64)


def official_crop_scale_frames(frames: np.ndarray, ratio: float) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    result = frames.copy()
    valid = frames[:, :, 2] != 0
    count = valid.sum(axis=1)
    xs = frames[:, :, 0]
    ys = frames[:, :, 1]
    xmin = np.min(np.where(valid, xs, np.inf), axis=1)
    xmax = np.max(np.where(valid, xs, -np.inf), axis=1)
    ymin = np.min(np.where(valid, ys, np.inf), axis=1)
    ymax = np.max(np.where(valid, ys, -np.inf), axis=1)
    scale = np.maximum(xmax - xmin, ymax - ymin) * np.float32(ratio)
    good = (count >= 4) & np.isfinite(scale) & (scale > 0)
    safe = np.where(good, scale, 1.0).astype(np.float32)
    origin_x = ((xmin + xmax - safe) * 0.5).astype(np.float32)
    origin_y = ((ymin + ymax - safe) * 0.5).astype(np.float32)
    result[:, :, 0] = ((frames[:, :, 0] - origin_x[:, None]) / safe[:, None] - 0.5) * 2.0
    result[:, :, 1] = ((frames[:, :, 1] - origin_y[:, None]) / safe[:, None] - 0.5) * 2.0
    result = np.clip(result, -1.0, 1.0)
    result[~good] = 0.0
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite normalized skeleton")
    return np.ascontiguousarray(result, dtype=np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    result = exponent / exponent.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite softmax")
    return result


def mean_clip_probability(frame_logits: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    frame_probability = _softmax(frame_logits)
    output = np.empty((len(offsets) - 1, CLASSES), dtype=np.float64)
    for row, (left, right) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        output[row] = frame_probability[int(left) : int(right)].mean(axis=0)
    output /= output.sum(axis=1, keepdims=True)
    return output


def _split_mean(values: np.ndarray, bins: int) -> list[np.ndarray]:
    return [
        values[index].mean(axis=0) if len(index) else np.zeros(values.shape[1], dtype=np.float64)
        for index in np.array_split(np.arange(len(values)), bins)
    ]


def sequence_raw_features(
    hidden: np.ndarray,
    offsets: np.ndarray,
    direct_probability: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row, (left_raw, right_raw) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        left, right = int(left_raw), int(right_raw)
        values = hidden[left:right].astype(np.float64)
        if len(values) != int(counts[row]) or not len(values):
            raise RuntimeError(f"invalid hidden sequence offset at row {row}")
        difference = np.abs(np.diff(values, axis=0))
        if not len(difference):
            difference = np.zeros((1, values.shape[1]), dtype=np.float64)
        rows.append(
            np.concatenate(
                [
                    values.mean(axis=0),
                    values.std(axis=0),
                    values.max(axis=0),
                    *_split_mean(values, 2),
                    *_split_mean(values, 3),
                    difference.mean(axis=0),
                    difference.std(axis=0),
                    direct_probability[row],
                    np.asarray([np.log1p(counts[row])], dtype=np.float64),
                ]
            )
        )
    output = np.asarray(rows, dtype=np.float32)
    if output.shape != (len(counts), 2601) or not np.isfinite(output).all():
        raise RuntimeError(f"row-local temporal feature closure mismatch: {output.shape}")
    return output


def signed_hash_projection(features: np.ndarray, output_dimension: int) -> np.ndarray:
    if output_dimension not in FEATURE_DIMENSIONS:
        raise RuntimeError("projection is outside the prelocked escalation")
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2601:
        raise RuntimeError("raw feature dimension mismatch")
    rng = np.random.default_rng(SEED)
    buckets = rng.integers(0, output_dimension, size=features.shape[1], dtype=np.int64)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=features.shape[1])
    output = np.zeros((len(features), output_dimension), dtype=np.float32)
    for source, (bucket, sign) in enumerate(zip(buckets.tolist(), signs.tolist(), strict=True)):
        output[:, int(bucket)] += np.float32(sign) * features[:, source]
    output *= np.float32(np.sqrt(output_dimension / features.shape[1]))
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite signed-hash projection")
    return output


def build_dstformer_model(official_root: Path, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    official_root = official_root.resolve()
    model_directory = official_root / "model" if (official_root / "model" / "DSTformer.py").is_file() else official_root
    required = {name: model_directory / name for name in ("DSTformer.py", "model_action.py", "drop.py")}
    if not all(path.is_file() for path in required.values()):
        raise RuntimeError("public model source closure mismatch")
    package = types.ModuleType("model")
    package.__path__ = [str(model_directory)]
    sys.modules["model"] = package
    loaded: dict[str, Any] = {}
    for short_name, file_name in (("drop", "drop.py"), ("DSTformer", "DSTformer.py"), ("model_action", "model_action.py")):
        full_name = f"model.{short_name}"
        spec = importlib.util.spec_from_file_location(full_name, required[file_name])
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load public source: {file_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        loaded[short_name] = module
    dst_module = loaded["DSTformer"]
    action_module = loaded["model_action"]
    backbone = dst_module.DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=64,
        dim_rep=64,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        maxlen=1,
        num_joints=17,
    )
    model = action_module.ActionNet(
        backbone=backbone,
        dim_rep=64,
        num_classes=CLASSES,
        dropout_ratio=0.5,
        version="class",
        hidden_dim=256,
        num_joints=17,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"DSTFormer state mismatch: {missing}, {unexpected}")
    model.eval()
    if (
        sum(parameter.numel() for parameter in model.parameters()) != 2_456_365
        or sum(value.numel() for value in model.state_dict().values()) != EXPECTED_DST_STATE_ELEMENTS
        or int(model.head.fc1.out_features) != 2048
    ):
        raise RuntimeError("DSTFormer architecture closure drifted")
    return model


def extract_official_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    archive = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(archive, dict) or "model" not in archive:
        raise RuntimeError("public checkpoint schema mismatch")
    state = {key.removeprefix("module."): value.detach().cpu().contiguous() for key, value in archive["model"].items()}
    if (
        len(state) != 69
        or sum(int(value.numel()) for value in state.values()) != EXPECTED_DST_STATE_ELEMENTS
        or state_dict_sha256(state) != EXPECTED_DST_STATE_SHA256
    ):
        raise RuntimeError("public checkpoint tensor inventory mismatch")
    return state


def extract_raw_features(
    model: torch.nn.Module,
    data: np.ndarray,
    counts: np.ndarray,
    frame_batch_size: int = DEFAULT_FRAME_BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.sparse import csr_matrix

    if frame_batch_size <= 0:
        raise RuntimeError("frame batch size must be positive")
    frames, offsets = flatten_valid(data, counts)
    hidden_dimension = int(model.head.fc1.out_features)
    rng = np.random.default_rng(FRAME_SEED)
    buckets = rng.integers(0, FRAME_PROJECTED_DIMENSION, size=hidden_dimension, dtype=np.int64)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=hidden_dimension)
    scale = np.float32(np.sqrt(FRAME_PROJECTED_DIMENSION / hidden_dimension))
    projector = csr_matrix(
        (signs * scale, (np.arange(hidden_dimension, dtype=np.int64), buckets)),
        shape=(hidden_dimension, FRAME_PROJECTED_DIMENSION),
        dtype=np.float32,
    )
    hidden_total = np.zeros((len(frames), FRAME_PROJECTED_DIMENSION), dtype=np.float32)
    logits_per_view: list[np.ndarray] = []
    with torch.inference_mode():
        for ratio in VIEWS:
            prepared = official_crop_scale_frames(frames, ratio)
            view_logits = np.empty((len(prepared), CLASSES), dtype=np.float32)
            for start in range(0, len(prepared), frame_batch_size):
                stop = min(start + frame_batch_size, len(prepared))
                tensor = torch.from_numpy(prepared[start:stop, None, None])
                n, m, t, joints, channels = tensor.shape
                representation = model.backbone.get_representation(
                    tensor.reshape(n * m, t, joints, channels)
                ).reshape(n, m, t, joints, -1)
                pooled = representation.permute(0, 1, 3, 4, 2).mean(dim=-1).reshape(n, m, -1).mean(dim=1)
                hidden = model.head.relu(model.head.bn(model.head.fc1(pooled)))
                view_logits[start:stop] = model.head.fc2(hidden).cpu().numpy().astype(np.float32, copy=False)
                hidden_array = hidden.cpu().numpy().astype(np.float32, copy=False)
                hidden_total[start:stop] += np.asarray(projector.T.dot(hidden_array.T).T, dtype=np.float32)
            logits_per_view.append(view_logits)
    hidden_total /= np.float32(len(VIEWS))
    collapsed_logits = np.mean(np.stack(logits_per_view), axis=0)
    direct = mean_clip_probability(collapsed_logits, offsets)
    raw = sequence_raw_features(hidden_total, offsets, direct, counts)
    return raw, {
        "rows": len(data),
        "valid_frames": len(frames),
        "frame_batch_size": frame_batch_size,
        "direct_argmax_sha256": array_sha256(direct.argmax(1).astype(np.int16)),
        "raw_features_sha256": array_sha256(raw),
    }


def extract_features(
    model: torch.nn.Module,
    data: np.ndarray,
    counts: np.ndarray,
    feature_dimension: int,
    frame_batch_size: int = DEFAULT_FRAME_BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw, report = extract_raw_features(model, data, counts, frame_batch_size)
    projected = signed_hash_projection(raw, feature_dimension)
    report = dict(report)
    report.update({"feature_dimension": feature_dimension, "features_sha256": array_sha256(projected)})
    return projected, report


def linear_decision(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    dimension = features.shape[1] if features.ndim == 2 else -1
    if (
        dimension not in FEATURE_DIMENSIONS
        or mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or coefficient.shape != (CLASSES, dimension)
        or intercept.shape != (CLASSES,)
        or np.any(scale <= 0)
    ):
        raise RuntimeError("linear student shape closure mismatch")
    result = ((features - mean) / scale) @ coefficient.T + intercept
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite decisions")
    return result


def build_deployable_payload(
    dst_state: dict[str, torch.Tensor],
    feature_dimension: int,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "contract": canonical_contract(feature_dimension),
        "dstformer_state": {key: value.detach().cpu().contiguous() for key, value in dst_state.items()},
        "feature_mean": torch.as_tensor(np.asarray(mean, dtype=np.float64)),
        "feature_scale": torch.as_tensor(np.asarray(scale, dtype=np.float64)),
        "ridge_coefficient": torch.as_tensor(np.asarray(coefficient, dtype=np.float64)),
        "ridge_intercept": torch.as_tensor(np.asarray(intercept, dtype=np.float64)),
        "class_order": torch.arange(CLASSES, dtype=torch.int64),
        "source_hashes": dict(sorted(source_hashes.items())),
    }


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def audit_deployable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "contract",
        "dstformer_state",
        "feature_mean",
        "feature_scale",
        "ridge_coefficient",
        "ridge_intercept",
        "class_order",
        "source_hashes",
    }
    if set(payload) != required or payload.get("schema") != SCHEMA:
        raise RuntimeError("deployable checkpoint schema mismatch")
    contract = payload["contract"]
    dimension = int(contract.get("feature_dimension", -1)) if isinstance(contract, dict) else -1
    if contract != canonical_contract(dimension):
        raise RuntimeError("deployable contract mismatch")
    state = payload["dstformer_state"]
    if not isinstance(state, dict) or len(state) != 69 or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("DSTFormer state closure mismatch")
    if sum(int(value.numel()) for value in state.values()) != EXPECTED_DST_STATE_ELEMENTS:
        raise RuntimeError("DSTFormer state element mismatch")
    if state_dict_sha256(state) != EXPECTED_DST_STATE_SHA256:
        raise RuntimeError("DSTFormer state digest mismatch")
    expected_shapes = {
        "feature_mean": (dimension,),
        "feature_scale": (dimension,),
        "ridge_coefficient": (CLASSES, dimension),
        "ridge_intercept": (CLASSES,),
        "class_order": (CLASSES,),
    }
    tensors = dict(state)
    for name, shape in expected_shapes.items():
        value = payload[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"deployable tensor shape mismatch: {name}")
        tensors[name] = value
    if not torch.equal(payload["class_order"], torch.arange(CLASSES, dtype=torch.int64)):
        raise RuntimeError("class order mismatch")
    if any(any(length in value.shape for length in (405, 2931)) or value.numel() in (405, 2931) for value in tensors.values()):
        raise RuntimeError("cohort-sized tensor is forbidden")
    if any(not torch.isfinite(value).all() for value in tensors.values() if value.is_floating_point()):
        raise RuntimeError("non-finite deployable tensor")
    forbidden = ("sm_test", "start_time", "session", "sample_id", "row_id", "feature_bank", "prototype", "exemplar")
    strings = [value.lower() for value in _walk_strings(payload)]
    hits = [fragment for fragment in forbidden if any(fragment in value for value in strings)]
    if hits:
        raise RuntimeError(f"forbidden checkpoint string: {hits}")
    return {
        "schema": "cuhkx/smalltrack/dst-ridge-rowlocal-checkpoint-audit-v2",
        "feature_dimension": dimension,
        "dstformer_state_tensors": len(state),
        "dstformer_state_elements": EXPECTED_DST_STATE_ELEMENTS,
        "dstformer_state_sha256": EXPECTED_DST_STATE_SHA256,
        "cohort_sized_tensor_present": False,
        "forbidden_string_present": False,
        "weights_only_load": True,
    }


def audit_deployable_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    result = audit_deployable_payload(payload)
    size = path.stat().st_size
    if size >= 100_000_000:
        raise RuntimeError("deployable checkpoint exceeds 100 MB")
    raw = path.read_bytes().lower()
    byte_forbidden = (b"sm_test", b"start_time", b"session", b"sample_id", b"row_id", b"feature_bank", b"prototype", b"exemplar")
    hits = [fragment.decode("ascii") for fragment in byte_forbidden if fragment in raw]
    if hits:
        raise RuntimeError(f"forbidden serialized checkpoint bytes: {hits}")
    result.update({"bytes": size, "sha256": sha256_file(path), "under_100mb": True, "byte_scan_hits": []})
    return result


def save_deployable_checkpoint(
    path: Path,
    dst_state: dict[str, torch.Tensor],
    feature_dimension: int,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_deployable_payload(dst_state, feature_dimension, mean, scale, coefficient, intercept, source_hashes),
        path,
    )
    return audit_deployable_checkpoint(path)


def load_deployable_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = audit_deployable_checkpoint(path)
    return torch.load(path, map_location="cpu", weights_only=True), audit
