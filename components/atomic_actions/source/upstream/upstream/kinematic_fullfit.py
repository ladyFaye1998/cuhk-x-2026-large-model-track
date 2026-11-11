"""Recovered kinematic training with explicit observation/output paths."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
N_CLASSES = 40
EPOCHS = 44
BATCH_SIZE = 128
SEED = 20260831
SKELETON_OFFSET = 95 + 28
POSE_WIDTH = 17 * 3
BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

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
    started = time.time()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    seed_all(SEED)
    bundle = np.load(find_one('small_sensor_features.npz'), allow_pickle=False)
    raw_train = kinematic_features(bundle['train_segments'].astype(np.float32))
    raw_test = kinematic_features(bundle['test_segments'].astype(np.float32))
    train_valid = bundle['train_masks'][:, 2:3].astype(np.float32)
    test_valid = bundle['test_masks'][:, 2:3].astype(np.float32)
    labels = bundle['train_labels'].astype(np.int64)
    mean = raw_train.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(raw_train.std(axis=(0, 1), keepdims=True), 0.0001)
    normalized_train = np.clip((raw_train - mean) / std, -7.0, 7.0).astype(np.float32)
    normalized_test = np.clip((raw_test - mean) / std, -7.0, 7.0).astype(np.float32)
    x_train = torch.from_numpy(normalized_train)
    y_train = torch.from_numpy(labels)
    v_train = torch.from_numpy(train_valid)
    x_test = torch.from_numpy(normalized_test)
    v_test = torch.from_numpy(test_valid)
    loader = DataLoader(TensorDataset(x_train, v_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(SEED + 80))
    model = SkeletonKinematicNet(raw_train.shape[-1])
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    weights = np.sqrt(np.max(counts) / np.maximum(counts, 1.0))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), label_smoothing=0.035)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0018, weight_decay=0.0015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=4e-05)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
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
            total_loss += float(loss) * len(sequence)
        scheduler.step()
        if (epoch + 1) % 4 == 0 or epoch + 1 == EPOCHS:
            print(json.dumps({'epoch': epoch + 1, 'train_loss': total_loss / len(labels), 'elapsed_seconds': time.time() - started}), flush=True)
    test = probabilities(model, x_test, v_test).astype(np.float32)
    state = {'schema_version': 'skeleton-kinematic-fullfit/v1', 'state_dict': model.state_dict(), 'normalization_mean': torch.from_numpy(mean), 'normalization_std': torch.from_numpy(std), 'epochs': EPOCHS, 'seed': SEED, 'channels': int(raw_train.shape[-1])}
    checkpoint_path = OUTPUT_ROOT / 'skeleton_kinematic_fullfit.pt'
    torch.save(state, checkpoint_path)
    np.savez_compressed(OUTPUT_ROOT / 'skeleton_kinematic_fullfit_predictions.npz', test=test)
    parameter_bytes = int(sum((parameter.numel() for parameter in model.parameters())) * 4)
    report = {'status': 'PASS', 'training_protocol': 'all official training users; architecture and hyperparameters frozen from six-fold OOF', 'epoch_policy': 'Fixed historical 44-epoch full-fit recipe; not reselected from the folds executed in this run', 'epochs': EPOCHS, 'seed': SEED, 'model_parameters': int(sum((parameter.numel() for parameter in model.parameters()))), 'model_parameter_bytes_float32': parameter_bytes, 'checkpoint_bytes': int(checkpoint_path.stat().st_size), 'test_rows': int(len(test)), 'test_prediction_classes': int(np.unique(test.argmax(axis=1)).size), 'test_ground_truth_used': False, 'elapsed_seconds': time.time() - started}
    assert report['checkpoint_bytes'] < 5000000
    (OUTPUT_ROOT / 'skeleton_kinematic_fullfit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
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
