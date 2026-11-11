"""Unchanged fitting and preprocessing functions extracted from all six recovered executors.

Extraction receipt binds their common AST to the original source files. The
portable entrypoint calls these functions with explicit observation assets.
"""
from __future__ import annotations
import hashlib, io, json, math, os, random
import numpy as np
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 20_260_822


FRAMES = 64


EPOCHS = 20


BATCH_SIZE = 32


BASE_LR = 0.03


WARMUP_EPOCHS = 2


MOMENTUM = 0.9


WEIGHT_DECAY = 0.0001


H36M17_INWARD = (
    (1, 0), (2, 1), (3, 2),
    (4, 0), (5, 4), (6, 5),
    (7, 0), (8, 7), (9, 8), (10, 9),
    (11, 8), (12, 11), (13, 12),
    (14, 8), (15, 14), (16, 15),
)


H36M17_PARENT = np.asarray([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])


H36M17_PARTS = (
    np.asarray([11, 12, 13], dtype=np.int64),
    np.asarray([14, 15, 16], dtype=np.int64),
    np.asarray([4, 5, 6], dtype=np.int64),
    np.asarray([1, 2, 3], dtype=np.int64),
    np.asarray([0, 7, 8, 9, 10], dtype=np.int64),
)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def h36m17_adjacency(max_hop: int = 2) -> np.ndarray:
    num_node = 17
    binary = np.zeros((num_node, num_node), dtype=np.float64)
    for index in range(num_node):
        binary[index, index] = 1.0
    for child, parent in H36M17_INWARD:
        binary[child, parent] = 1.0
        binary[parent, child] = 1.0
    hop = np.full((num_node, num_node), np.inf, dtype=np.float64)
    arrivals = np.stack([
        np.linalg.matrix_power(binary, distance) for distance in range(max_hop + 1)
    ]) > 0
    for distance in range(max_hop, -1, -1):
        hop[arrivals[distance]] = distance
    adjacency = np.zeros_like(binary)
    for distance in range(max_hop + 1):
        adjacency[hop == distance] = 1.0
    degree = adjacency.sum(axis=0)
    inverse_degree = np.zeros_like(adjacency)
    positive = degree > 0
    inverse_degree[positive, positive] = degree[positive] ** -1
    normalized = adjacency @ inverse_degree
    output = np.zeros((max_hop + 1, num_node, num_node), dtype=np.float32)
    for distance in range(max_hop + 1):
        output[distance][hop == distance] = normalized[hop == distance]
    return output


def safe_load_official_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    unsafe = sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    expected = [
        "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy.dtype",
        "numpy.ndarray",
    ]
    if unsafe != expected:
        raise RuntimeError(f"official checkpoint global inventory drift: {unsafe}")
    numpy_core = getattr(np, "_core", np.core)
    safe_globals = [
        (numpy_core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        (np.ndarray, "numpy.ndarray"),
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise RuntimeError("checkpoint does not contain official model state")
    output: dict[str, torch.Tensor] = {}
    for raw_name, tensor in raw["model"].items():
        if not isinstance(raw_name, str) or not isinstance(tensor, torch.Tensor):
            raise RuntimeError("non-tensor official model entry")
        name = raw_name.removeprefix("module.")
        if name in output:
            raise RuntimeError(f"duplicate checkpoint key: {name}")
        output[name] = tensor.detach().cpu()
    return output


def exclusion_reason(name: str) -> str | None:
    if name.startswith("classifier."):
        return "classifier_60_to_40_reset"
    if name.endswith(".A"):
        return "ntu25_adjacency_replaced_by_h36m17"
    if name.endswith(".edge"):
        return "ntu25_edge_importance_reinitialized_for_h36m17"
    return None


def apply_transfer(model: nn.Module, source: dict[str, torch.Tensor]) -> dict[str, object]:
    target = model.state_dict()
    target_parameters = dict(model.named_parameters())
    loaded: dict[str, torch.Tensor] = {}
    records: list[dict[str, object]] = []
    unexpected: list[str] = []
    forbidden_mismatch: list[str] = []
    for name in sorted(source):
        tensor = source[name]
        reason = exclusion_reason(name)
        if name not in target:
            unexpected.append(name)
            continue
        if reason is not None:
            records.append({"key": name, "action": "reinitialize", "reason": reason,
                            "source_shape": list(tensor.shape), "target_shape": list(target[name].shape)})
        elif tensor.shape != target[name].shape:
            forbidden_mismatch.append(name)
        else:
            loaded[name] = tensor.to(dtype=target[name].dtype)
            records.append({"key": name, "action": "load_exact", "shape": list(tensor.shape)})
    missing, post_unexpected = model.load_state_dict(loaded, strict=False)
    allowed_missing = {name for name in target if exclusion_reason(name) is not None}
    if set(missing) != allowed_missing or unexpected or post_unexpected or forbidden_mismatch:
        raise RuntimeError("deterministic partial-load rule failed")
    total = sum(parameter.numel() for parameter in target_parameters.values())
    loaded_elements = sum(target_parameters[name].numel() for name in loaded if name in target_parameters)
    if loaded_elements / total < 0.95:
        raise RuntimeError("predeclared 95% parameter-coverage gate failed")
    return {
        "source_state_tensors": len(source),
        "loaded_state_tensors": len(loaded),
        "reinitialized_state_tensors": len(allowed_missing),
        "target_parameter_elements": total,
        "loaded_parameter_elements": loaded_elements,
        "parameter_element_coverage": loaded_elements / total,
        "records": records,
    }


def temporal_sample(sequence: np.ndarray, valid: int, training: bool, seed: int) -> np.ndarray:
    sequence = np.asarray(sequence[:valid], dtype=np.float32)
    if valid <= 1:
        return np.repeat(sequence[:1], FRAMES, axis=0)
    rng = np.random.default_rng(seed)
    if training:
        trim_left = float(rng.uniform(0.0, min(2.0, (valid - 1) * 0.08)))
        trim_right = float(rng.uniform(0.0, min(2.0, (valid - 1) * 0.08)))
    else:
        trim_left = trim_right = 0.0
    positions = np.linspace(trim_left, valid - 1 - trim_right, FRAMES, dtype=np.float64)
    if training:
        offsets = rng.uniform(-0.35, 0.35, size=FRAMES)
        offsets[0] = offsets[-1] = 0.0
        positions = np.maximum.accumulate(np.clip(positions + offsets, trim_left, valid - 1 - trim_right))
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, valid - 1)
    alpha = (positions - lower).astype(np.float32)[:, None, None]
    return sequence[lower] * (1.0 - alpha) + sequence[upper] * alpha


def normalize_and_augment(sequence: np.ndarray, training: bool, seed: int) -> np.ndarray:
    sequence = sequence.astype(np.float32, copy=True)
    sequence -= sequence[0:1, 0:1, :]
    pose = sequence - sequence[:, 0:1, :]
    rms = float(np.sqrt(np.mean(np.square(pose[:, 1:, :]), dtype=np.float64)))
    if not np.isfinite(rms) or rms < 1e-6:
        rms = 1.0
    sequence /= rms
    if training:
        rng = np.random.default_rng(seed ^ 0x5EEDB4)
        angle = float(rng.uniform(-math.pi / 18.0, math.pi / 18.0))
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation_y = np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=np.float32
        )
        sequence = sequence @ rotation_y.T
        sequence *= np.float32(rng.uniform(0.9, 1.1))
        sequence += rng.normal(0.0, 0.005, size=sequence.shape).astype(np.float32)
    return sequence


def multi_input(sequence: np.ndarray) -> np.ndarray:
    data = sequence.transpose(2, 0, 1)[:, :, :, None]
    channels, frames, joints, persons = data.shape
    joint = np.zeros((channels * 2, frames, joints, persons), dtype=np.float32)
    velocity = np.zeros_like(joint)
    bone = np.zeros_like(joint)
    joint[:channels] = data
    joint[channels:] = data - data[:, :, 0:1, :]
    velocity[:channels, :-1] = data[:, 1:] - data[:, :-1]
    velocity[channels:, :-2] = data[:, 2:] - data[:, :-2]
    for joint_index, parent_index in enumerate(H36M17_PARENT):
        bone[:channels, :, joint_index] = data[:, :, joint_index] - data[:, :, parent_index]
    bone_length = np.sqrt(np.sum(np.square(bone[:channels]), axis=0)) + np.float32(1e-4)
    bone[channels:] = np.arccos(np.clip(bone[:channels] / bone_length[None], -1.0, 1.0))
    output = np.stack((joint, velocity, bone), axis=0)
    if output.shape != (3, 6, FRAMES, 17, 1) or not np.isfinite(output).all():
        raise RuntimeError("JVB preprocessing failed")
    return output


class SkeletonDataset(Dataset):
    def __init__(self, skeleton: np.ndarray, valid: np.ndarray, labels: np.ndarray,
                 indices: np.ndarray, training: bool):
        self.skeleton = skeleton
        self.valid = valid
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, local_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_index = int(self.indices[local_index])
        item_seed = SEED + self.epoch * 1_000_003 + source_index * 101
        sequence = temporal_sample(
            self.skeleton[source_index], int(self.valid[source_index]), self.training, item_seed
        )
        sequence = normalize_and_augment(sequence, self.training, item_seed)
        features = multi_input(sequence)
        return (
            torch.from_numpy(features),
            torch.tensor(int(self.labels[source_index]), dtype=torch.long),
            torch.tensor(source_index, dtype=torch.long),
        )


def epoch_learning_rate(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return BASE_LR * float(epoch + 1) / WARMUP_EPOCHS
    progress = float(epoch + 1 - WARMUP_EPOCHS) / float(EPOCHS - WARMUP_EPOCHS)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(dataset: SkeletonDataset, epoch: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED + epoch * 65_537)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=False,
    )


def train_arm(name: str, model: nn.Module, train_dataset: SkeletonDataset,
              held_dataset: SkeletonDataset, device: torch.device) -> tuple[dict[str, object], np.ndarray, dict[str, torch.Tensor]]:
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    for epoch in range(EPOCHS):
        train_dataset.set_epoch(epoch)
        learning_rate = epoch_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, labels, _ in make_loader(train_dataset, epoch, shuffle=True):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_examples += labels.numel()
        history.append({
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "training_cross_entropy": total_loss / total_examples,
        })

    held_dataset.set_epoch(0)
    model.eval()
    held_logits: list[np.ndarray] = []
    held_labels: list[np.ndarray] = []
    held_indices: list[np.ndarray] = []
    with torch.inference_mode():
        for features, labels, indices in make_loader(held_dataset, 0, shuffle=False):
            logits, _ = model(features.to(device, non_blocking=True))
            held_logits.append(logits.cpu().numpy())
            held_labels.append(labels.numpy())
            held_indices.append(indices.numpy())
    logits_array = np.concatenate(held_logits)
    labels_array = np.concatenate(held_labels)
    indices_array = np.concatenate(held_indices)
    order = np.argsort(indices_array)
    logits_array = logits_array[order]
    labels_array = labels_array[order]
    indices_array = indices_array[order]
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    cross_entropy = float(nn.functional.cross_entropy(
        torch.from_numpy(logits_array), torch.from_numpy(labels_array)
    ))
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    summary = {
        "name": name,
        "epochs": EPOCHS,
        "history": history,
        "held_examples": int(labels_array.size),
        "held_top1_accuracy": float(np.mean(predictions == labels_array)),
        "held_cross_entropy": cross_entropy,
        "held_indices_sha256": hashlib.sha256(indices_array.astype("<i8").tobytes()).hexdigest(),
    }
    return summary, probabilities, final_state


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    import zipfile

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def serialize_half_checkpoint(state: dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    frozen = {
        key: (value.half() if value.is_floating_point() else value)
        for key, value in sorted(state.items())
    }
    torch.save({"model": frozen, "architecture": "EfficientGCN-B4-SG-H36M17-40"}, buffer)
    return buffer.getvalue()
