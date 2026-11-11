import hashlib
"""Recovered kinematic training with explicit observation/output paths."""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
FOLDS = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [16, 17, 18], [19, 20, 21], [22, 23, 24]]
N_CLASSES = 40
EPOCHS = 72
BATCH_SIZE = 128
SKELETON_OFFSET = 95 + 28
POSE_WIDTH = 17 * 3
BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# Fingerprints of official observation arrays, independent of ZIP headers,
# compression-library details and the operating system that writes the NPZ.
EXPECTED_OBSERVATION_ARRAYS = {
    'train_segments': ((2931, 16, 430), '<f4', '29e2f93df383d0589ef2fb84cbcbf88b02b6f3d6ad5abc2388a5813ab8ac31ae'),
    'train_global': ((2931, 1723), '<f4', '248a3605bc96cf9733408f87195584ed71d1844c3ba802bd8a5431586ee428a1'),
    'train_masks': ((2931, 3), '|u1', '270e43fb85f3f61dc3117eb78fcd067147572e925f4bb59bd6c45f63b832409b'),
    'train_labels': ((2931,), '<i8', 'c0eb91cd62eb7b4710e33dc56bbfcc1a4399088d089e8710c3846fc53159e553'),
    'train_users': ((2931,), '<i8', 'dbc919d48023feccdfc21e88dd4424e28a0582303c352ccb69ce4e25ebfc5d5b'),
    'train_trials': ((2931,), '<U5', '13e5d384f967cf8a0b2ba4f28998f3690f651989542b2e215632fe7f87b612ca'),
    'test_segments': ((405, 16, 430), '<f4', '13a614d797b23f63fca0d7044430e58fbb7235c5a236690fc9f0fb0787cbbc6f'),
    'test_global': ((405, 1723), '<f4', 'edab5b2f72b23bf81ec4edcbada126bb4f6b4b26eedd2cf50f458ea2fcbe4f45'),
    'test_masks': ((405, 3), '|u1', '6aa2de98017b00c832290c9c4ccd040323d8b9f749d019a9a5e2b73a8239c9fe'),
    'test_numbers': ((405,), '<i8', '3c78db76e2411881e8a811db66893c1b83de291214d5e4291bc1905ddd58ffcb'),
}

def validate_observation_arrays(bundle):
    if set(bundle.files) != set(EXPECTED_OBSERVATION_ARRAYS):
        raise ValueError('Official observation array names differ')
    for name, (shape, dtype, digest) in EXPECTED_OBSERVATION_ARRAYS.items():
        value = bundle[name]
        if value.shape != shape or value.dtype.str != dtype:
            raise ValueError('Official observation shape or dtype differs: ' + name)
        if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != digest:
            raise ValueError('Official observation content differs: ' + name)

def validate_observation_archive(path):
    with np.load(path, allow_pickle=False) as bundle:
        validate_observation_arrays(bundle)

def find_one(pattern):
    if pattern != 'small_sensor_features.npz':
        raise ValueError('Unexpected observation input pattern')
    return FEATURE_PATH

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def kinematic_features(segments):
    skeleton = segments[:, :, SKELETON_OFFSET:]
    position = skeleton[:, :, POSE_WIDTH:2 * POSE_WIDTH].reshape(len(skeleton), 16, 17, 3)
    spread = skeleton[:, :, 3 * POSE_WIDTH:4 * POSE_WIDTH].reshape(len(skeleton), 16, 17, 3)
    intrabin_motion = skeleton[:, :, 5 * POSE_WIDTH:6 * POSE_WIDTH].reshape(len(skeleton), 16, 17, 3)
    velocity = np.diff(position, axis=1, prepend=position[:, :1])
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1])
    bone_vectors = np.stack([position[:, :, child] - position[:, :, parent] for parent, child in BONES], axis=2)
    bone_lengths = np.linalg.norm(bone_vectors, axis=3)
    root_speed = np.linalg.norm(velocity[:, :, [0, 5, 6, 11, 12]], axis=3)
    frame_count = skeleton[:, :, -1:]
    output = np.concatenate([position.reshape(len(skeleton), 16, -1), spread.reshape(len(skeleton), 16, -1), intrabin_motion.reshape(len(skeleton), 16, -1), velocity.reshape(len(skeleton), 16, -1), acceleration.reshape(len(skeleton), 16, -1), bone_vectors.reshape(len(skeleton), 16, -1), bone_lengths, root_speed, frame_count], axis=2)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

class ResidualTCN(nn.Module):

    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation), nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(0.12), nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation), nn.GroupNorm(8, width))
        self.activation = nn.GELU()

    def forward(self, values):
        return self.activation(values + self.net(values))

class SkeletonKinematicNet(nn.Module):

    def __init__(self, channels, width=160, hidden=128):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(channels, width), nn.LayerNorm(width), nn.GELU())
        self.tcn = nn.Sequential(ResidualTCN(width, 1), ResidualTCN(width, 2), ResidualTCN(width, 4))
        self.gru = nn.GRU(width, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden * 2, 96), nn.Tanh(), nn.Linear(96, 1))
        self.head = nn.Sequential(nn.Linear(hidden * 2 * 3 + 1, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.28), nn.Linear(256, N_CLASSES))

    def forward(self, sequence, valid):
        values = self.input(sequence)
        values = self.tcn(values.transpose(1, 2)).transpose(1, 2)
        values, _ = self.gru(values)
        attention = self.attention(values).softmax(dim=1)
        attended = (values * attention).sum(dim=1)
        pooled = torch.cat([attended, values.mean(dim=1), values.amax(dim=1), valid], dim=1)
        return self.head(pooled)

def probabilities(model, sequence, valid, batch_size=256):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(sequence), batch_size):
            logits = model(sequence[start:start + batch_size], valid[start:start + batch_size])
            output.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(output)

def main():
    torch.set_num_threads(4)
    print(json.dumps({'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu'}), flush=True)
    import hashlib
    feature_path = find_one('small_sensor_features.npz')
    validate_observation_archive(feature_path)
    assert not list(OUTPUT_ROOT.glob('fold_*.pt')), 'Existing fold checkpoints: inspect and resume explicitly; do not overwrite.'
    bundle = np.load(feature_path, allow_pickle=False)
    raw_train = kinematic_features(bundle['train_segments'].astype(np.float32))
    raw_test = kinematic_features(bundle['test_segments'].astype(np.float32))
    train_valid = bundle['train_masks'][:, 2:3].astype(np.float32)
    test_valid = bundle['test_masks'][:, 2:3].astype(np.float32)
    labels = bundle['train_labels'].astype(np.int64)
    users = bundle['train_users'].astype(np.int64)
    oof = np.zeros((len(labels), N_CLASSES), dtype=np.float32)
    test = np.zeros((len(raw_test), N_CLASSES), dtype=np.float64)
    fold_reports = []
    started = time.time()
    for fold, heldout_users in enumerate(FOLDS):
        seed_all(20260831 + fold)
        valid_mask = np.isin(users, heldout_users)
        train_mask = ~valid_mask
        mean = raw_train[train_mask].mean(axis=(0, 1), keepdims=True)
        std = np.maximum(raw_train[train_mask].std(axis=(0, 1), keepdims=True), 0.0001)
        normalized_train = np.clip((raw_train - mean) / std, -7.0, 7.0).astype(np.float32)
        normalized_test = np.clip((raw_test - mean) / std, -7.0, 7.0).astype(np.float32)
        x_train = torch.from_numpy(normalized_train[train_mask])
        y_train = torch.from_numpy(labels[train_mask])
        v_train = torch.from_numpy(train_valid[train_mask])
        x_valid = torch.from_numpy(normalized_train[valid_mask])
        v_valid = torch.from_numpy(train_valid[valid_mask])
        x_test = torch.from_numpy(normalized_test)
        v_test = torch.from_numpy(test_valid)
        loader = DataLoader(TensorDataset(x_train, v_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(20260911 + fold))
        model = SkeletonKinematicNet(raw_train.shape[-1])
        counts = np.bincount(labels[train_mask], minlength=N_CLASSES).astype(np.float32)
        weights = np.sqrt(np.max(counts) / np.maximum(counts, 1.0))
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), label_smoothing=0.035)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0018, weight_decay=0.0015)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=4e-05)
        best_state = None
        best_score = -1.0
        best_epoch = 0
        stale = 0
        for epoch in range(EPOCHS):
            model.train()
            for sequence, available, targets in loader:
                if epoch < EPOCHS - 10:
                    sequence = sequence + 0.01 * torch.randn_like(sequence)
                    temporal_drop = torch.rand((len(sequence), sequence.shape[1], 1)) < 0.025
                    sequence = sequence.masked_fill(temporal_drop, 0.0)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(sequence, available), targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(json.dumps({'fold_progress': fold, 'epoch': epoch + 1, 'elapsed_seconds': time.time() - started}), flush=True)
            if epoch >= 14 and (epoch + 1) % 2 == 0:
                valid_probs = probabilities(model, x_valid, v_valid)
                score = float(np.mean(valid_probs.argmax(axis=1) == labels[valid_mask]))
                if score > best_score:
                    best_score = score
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= 9:
                    break
        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        model.load_state_dict(best_state)
        torch.save({'state_dict': best_state, 'normalization_mean': torch.from_numpy(mean), 'normalization_std': torch.from_numpy(std), 'channels': int(raw_train.shape[-1]), 'held_users': heldout_users, 'fit_users': sorted(map(int, set(users[train_mask]))), 'best_epoch': int(best_epoch), 'seed': 20260831 + fold, 'source_sha256': 'afffb21785d5c547d46786093c03c9569276d7c156c23468f1b588429e0176b5', 'observation_sha256': hashlib.sha256(FEATURE_PATH.read_bytes()).hexdigest()}, OUTPUT_ROOT / f'fold_{fold}.pt')
        valid_probs = probabilities(model, x_valid, v_valid)
        oof[valid_mask] = valid_probs
        test += probabilities(model, x_test, v_test) / len(FOLDS)
        fold_report = {'fold': fold, 'heldout_users': heldout_users, 'valid_rows': int(valid_mask.sum()), 'best_epoch': best_epoch, 'accuracy': float(np.mean(valid_probs.argmax(axis=1) == labels[valid_mask])), 'elapsed_seconds': time.time() - started}
        np.savez_compressed(OUTPUT_ROOT / f'fold_{fold}_outputs.npz', valid_indices=np.flatnonzero(valid_mask), valid=valid_probs, test=probabilities(model, x_test, v_test))
        fold_reports.append(fold_report)
        (OUTPUT_ROOT / 'completed_folds.json').write_text(json.dumps(fold_reports, indent=2))
        print(json.dumps(fold_report), flush=True)
    report = {'validation_protocol': 'six disjoint held-out-user folds; official training labels only', 'supervision': 'official training labels', 'input': 'normalized 3D skeleton pose, spread, motion, velocity, acceleration, and bones', 'feature_shape': list(raw_train.shape), 'folds': fold_reports, 'oof_accuracy': float(np.mean(oof.argmax(axis=1) == labels)), 'oof_correct': int(np.sum(oof.argmax(axis=1) == labels)), 'model_parameters': int(sum((parameter.numel() for parameter in model.parameters()))), 'model_parameter_bytes_float32': int(sum((parameter.numel() for parameter in model.parameters())) * 4), 'elapsed_seconds': time.time() - started}
    np.savez_compressed(OUTPUT_ROOT / 'skeleton_kinematic_predictions.npz', labels=labels, users=users, oof=oof, test=test.astype(np.float32))
    (OUTPUT_ROOT / 'skeleton_kinematic_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    FEATURE_PATH = Path(args.features).resolve()
    OUTPUT_ROOT = Path(args.out).resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    main()
