"""Locked H36M17 -> SkateFormer NTU25 preprocessing.

This module contains no labels, metrics, test paths, candidate paths, or submission logic.
It is a deterministic adapter for the official SkateFormer evaluation path pinned by the
route protocols in this directory.  Shapes use the official conventions:

* incoming CUHK-X skeleton: ``[T, 17, 3]`` (one H36M17 actor);
* official raw NTU skeleton: ``[C, T, 25, 2]``;
* SkateFormer input after partition: ``[1, C, 64, 24, 2]``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
import torch.nn.functional as F


WINDOW: Final[int] = 64
EVAL_KEEP_FRACTION: Final[float] = 0.95

# H36M17 zero-based indices in the already-established CUHK-X skeleton order.
# Each tuple is (NTU25 zero-based destination, H36M17 zero-based sources).
H36M17_TO_NTU25: Final[tuple[tuple[int, tuple[int, ...]], ...]] = (
    (0, (0,)),    # spine base / pelvis
    (1, (7,)),    # spine mid
    (2, (9,)),    # neck
    (3, (10,)),   # head
    (4, (11,)),   # left shoulder
    (5, (12,)),   # left elbow
    (6, (13,)),   # left wrist
    (7, (13,)),   # left hand proxy
    (8, (14,)),   # right shoulder
    (9, (15,)),   # right elbow
    (10, (16,)),  # right wrist
    (11, (16,)),  # right hand proxy
    (12, (4,)),   # left hip
    (13, (5,)),   # left knee
    (14, (6,)),   # left ankle
    (15, (6,)),   # left foot proxy
    (16, (1,)),   # right hip
    (17, (2,)),   # right knee
    (18, (3,)),   # right ankle
    (19, (3,)),   # right foot proxy
    (20, (8,)),   # spine shoulder / thorax
    (21, (13,)),  # left hand tip proxy
    (22, (13,)),  # left thumb proxy
    (23, (16,)),  # right hand tip proxy
    (24, (16,)),  # right thumb proxy
)

# Byte-for-byte values from skateformer.preprocessing.NTU_PARTITION_INDEX.
NTU_PARTITION_INDEX: Final[np.ndarray] = np.asarray(
    [6, 7, 21, 22, 10, 11, 23, 24, 12, 13, 14, 15,
     16, 17, 18, 19, 4, 8, 5, 9, 1, 2, 0, 3],
    dtype=np.int64,
)

# Byte-for-byte values from skateformer.preprocessing.NTU_BONE_PAIRS.
NTU_BONE_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (1, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
    (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
    (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
    (19, 18), (20, 1), (21, 7), (22, 7), (23, 11), (24, 11),
)


def _require_h36m17(sequence: np.ndarray) -> np.ndarray:
    data = np.asarray(sequence)
    if data.ndim != 3 or data.shape[1:] != (17, 3):
        raise ValueError(f"expected [T,17,3], got {data.shape}")
    if data.shape[0] < 1:
        raise ValueError("a sequence must contain at least one valid frame")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"numeric skeleton required, got {data.dtype}")
    data = np.asarray(data, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("skeleton contains NaN or infinity")
    return data


def map_h36m17_to_ntu25(sequence: np.ndarray) -> np.ndarray:
    """Map ``[T,17,3]`` H36M17 joints to the fixed pseudo-NTU25 layout."""
    source = _require_h36m17(sequence)
    pseudo = np.empty((source.shape[0], 25, 3), dtype=np.float32)
    seen: set[int] = set()
    for destination, source_indices in H36M17_TO_NTU25:
        if destination in seen:
            raise AssertionError(f"duplicate mapping destination {destination}")
        seen.add(destination)
        pseudo[:, destination] = source[:, np.asarray(source_indices, dtype=np.int64)].mean(
            axis=1, dtype=np.float32
        )
    if seen != set(range(25)):
        raise AssertionError("mapping must populate every NTU25 joint exactly once")
    return pseudo


def source_native_translate_and_duplicate(pseudo_ntu25: np.ndarray) -> np.ndarray:
    """Apply official NTU ``seq_translation`` semantics, then duplicate one actor.

    The official NTU preprocessing subtracts actor 1's joint-2 coordinate from its
    first real frame from every valid coordinate.  Official ``align_frames`` duplicates
    a one-actor sequence into both person slots.  This function performs those steps on
    valid frames only and returns ``[C,T,25,2]``.
    """
    pseudo = np.asarray(pseudo_ntu25, dtype=np.float32)
    if pseudo.ndim != 3 or pseudo.shape[1:] != (25, 3) or pseudo.shape[0] < 1:
        raise ValueError(f"expected [T,25,3], got {pseudo.shape}")
    if not np.isfinite(pseudo).all():
        raise ValueError("pseudo-NTU skeleton contains NaN or infinity")
    origin = pseudo[0, 1].copy()  # official one-based joint 2
    translated = pseudo - origin[None, None, :]
    ctv = translated.transpose(2, 0, 1)
    return np.stack((ctv, ctv), axis=-1).astype(np.float32, copy=False)


def crop_resize_eval(
    raw_ntu: np.ndarray,
    *,
    window: int = WINDOW,
    keep_fraction: float = EVAL_KEEP_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact official Hub evaluation crop/resize path for valid-only frames."""
    data = np.asarray(raw_ntu, dtype=np.float32)
    if data.ndim != 4 or data.shape[0] != 3 or data.shape[2:] != (25, 2):
        raise ValueError(f"expected [3,T,25,2], got {data.shape}")
    valid_size = int(data.shape[1])
    if valid_size < 1:
        raise ValueError("no valid frame")
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(f"invalid keep fraction {keep_fraction}")
    bias = int((1.0 - keep_fraction) * valid_size / 2.0)
    c_b, c_e = bias, valid_size - bias
    if c_e <= c_b:
        raise ValueError("empty center crop")
    cropped = data[:, c_b:c_e]
    channels, _, joints, people = cropped.shape
    tensor = torch.from_numpy(np.ascontiguousarray(cropped)).float()
    tensor = tensor.permute(2, 3, 0, 1).contiguous().view(joints * people, channels, -1)
    tensor = F.interpolate(tensor, size=window, mode="linear", align_corners=False)
    tensor = tensor.contiguous().view(joints, people, channels, window)
    clip = tensor.permute(2, 3, 0, 1).contiguous().numpy()

    index_t = torch.arange(start=c_b, end=c_e, dtype=torch.float32)
    index_t = F.interpolate(
        index_t[None, None, :], size=window, mode="linear", align_corners=False
    ).squeeze(0).squeeze(0)
    index_t = 2.0 * index_t / float(valid_size) - 1.0
    return clip, index_t.numpy()


def joint_to_bone(clip: np.ndarray) -> np.ndarray:
    """Official NTU child-minus-parent bone transform on ``[C,T,25,2]``."""
    data = np.asarray(clip, dtype=np.float32)
    if data.ndim != 4 or data.shape[0] != 3 or data.shape[2:] != (25, 2):
        raise ValueError(f"expected [3,T,25,2], got {data.shape}")
    bone = np.zeros_like(data)
    for child, parent in NTU_BONE_PAIRS:
        bone[:, :, child] = data[:, :, child] - data[:, :, parent]
    return bone


def partition_joints(clip: np.ndarray) -> np.ndarray:
    """Apply the official 25-to-24 skeletal partition ordering."""
    data = np.asarray(clip, dtype=np.float32)
    if data.ndim != 4 or data.shape[0] != 3 or data.shape[2:] != (25, 2):
        raise ValueError(f"expected [3,T,25,2], got {data.shape}")
    return np.ascontiguousarray(data[:, :, NTU_PARTITION_INDEX, :])


def prepare_eval_pair(sequence: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return joint input, bone input, and shared ``index_t`` for one sequence."""
    pseudo = map_h36m17_to_ntu25(sequence)
    raw = source_native_translate_and_duplicate(pseudo)
    resized, index_t = crop_resize_eval(raw)
    joint = partition_joints(resized)
    bone = partition_joints(joint_to_bone(resized))
    joint_tensor = torch.from_numpy(joint).float().unsqueeze(0)
    bone_tensor = torch.from_numpy(bone).float().unsqueeze(0)
    time_tensor = torch.from_numpy(np.ascontiguousarray(index_t)).float().unsqueeze(0)
    expected = (1, 3, WINDOW, 24, 2)
    if tuple(joint_tensor.shape) != expected or tuple(bone_tensor.shape) != expected:
        raise AssertionError("unexpected SkateFormer tensor shape")
    if tuple(time_tensor.shape) != (1, WINDOW):
        raise AssertionError("unexpected index_t shape")
    return joint_tensor, bone_tensor, time_tensor


__all__ = [
    "EVAL_KEEP_FRACTION",
    "H36M17_TO_NTU25",
    "NTU_BONE_PAIRS",
    "NTU_PARTITION_INDEX",
    "WINDOW",
    "crop_resize_eval",
    "joint_to_bone",
    "map_h36m17_to_ntu25",
    "partition_joints",
    "prepare_eval_pair",
    "source_native_translate_and_duplicate",
]
