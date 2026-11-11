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
WINDOWS = 12
FRAMES = 16
SOURCE_SIZE = 128
VJEPA_SIZE = 256
VJEPA_RESIZE = 292
WINDOW_HALF_SPAN = 0.05
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
LETTERS = 'ABCD'
ACTIONS = ('Brushing teeth', 'Calling', 'Checking body temperature', 'Checking the time', 'Combing hair', 'Drinking', 'Eating', 'Folding clothes', 'Getting dressed', 'Grabbing utensils', 'Jumping jacks', 'Listening to the music with headphones', 'Lunges', 'Lying down', 'Massaging oneself', 'Mopping', 'Peeling fruit', 'Playing games', 'Pouring', 'Reading', 'Running', 'Sitting down', 'Squats', 'Standing up', 'Stirring', 'Stretching', 'Sweeping', 'Taking a selfie', 'Taking medicine', 'Turning a page', 'Typing on a keyboard', 'Undressing', 'Using a phone', 'Walking', 'Washing dishes', 'Washing face', 'Watching TV', 'Wiping hands', 'Wiping surface', 'Writing')
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}

class ExperimentError(RuntimeError):
    pass

def parse_actions(value: str) -> list[int]:
    names = [part.strip() for part in value.split(',')]
    if not names or any((name not in ACTION_INDEX for name in names)):
        raise ExperimentError(f'unknown action option: {value!r}')
    result = [ACTION_INDEX[name] for name in names]
    if len(set(result)) != len(result):
        raise ExperimentError(f'duplicate action in option: {value!r}')
    return result

def center_square_resize(rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2
    height, width = rgb.shape[:2]
    side = min(height, width)
    top, left = ((height - side) // 2, (width - side) // 2)
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
        raise ExperimentError(f'invalid video frame count: {total}')
    centers = np.linspace(0.04, 0.96, WINDOWS)
    result = []
    for center in centers:
        start, end = (max(0.0, center - WINDOW_HALF_SPAN), min(1.0, center + WINDOW_HALF_SPAN))
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
        raise ExperimentError(f'decoder missed requested frames in {path.name}: {missing[:8]}')
    frames = np.stack([[retained[int(value)] for value in row] for row in indices]).astype(np.uint8)
    if frames.shape != (WINDOWS, FRAMES, SOURCE_SIZE, SOURCE_SIZE, 3):
        raise ExperimentError(f'decoded shape drift: {frames.shape}')
    gray = frames.astype(np.float32).mean(axis=-1) / 255.0
    diffs = np.abs(np.diff(gray, axis=1))
    pace = np.stack([diffs.mean((1, 2, 3)), diffs.std((1, 2, 3)), np.quantile(diffs, 0.9, axis=(1, 2, 3)), gray.mean((1, 2, 3)), gray.std((1, 2, 3)), np.full(WINDOWS, math.log1p(total / fps) if fps > 0 else math.log1p(total), dtype=np.float32)], axis=1).astype(np.float32)
    return (frames, pace)

def extract_path_features(depth_frames: np.ndarray, ir_frames: np.ndarray, depth_pace: np.ndarray, ir_pace: np.ndarray, vjepa: Any, siglip: Any, sig_processor: Any, torch: Any) -> np.ndarray:
    import torch.nn.functional as functional
    vj_blocks = []
    with torch.inference_mode():
        for t in range(WINDOWS):
            raw = np.stack((depth_frames[t], ir_frames[t]))
            batch = torch.from_numpy(raw).permute(0, 1, 4, 2, 3).float().div_(255.0)
            flat = batch.reshape(2 * FRAMES, 3, SOURCE_SIZE, SOURCE_SIZE)
            flat = functional.interpolate(flat, size=(VJEPA_RESIZE, VJEPA_RESIZE), mode='bilinear', align_corners=False)
            offset = (VJEPA_RESIZE - VJEPA_SIZE) // 2
            batch = flat[:, :, offset:offset + VJEPA_SIZE, offset:offset + VJEPA_SIZE].reshape(2, FRAMES, 3, VJEPA_SIZE, VJEPA_SIZE)
            mean = torch.from_numpy(MEAN).view(1, 1, 3, 1, 1)
            std = torch.from_numpy(STD).view(1, 1, 3, 1, 1)
            batch = batch.sub(mean).div(std).to('cuda', dtype=torch.float16)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                encoded = vjepa.vjepa2(pixel_values_videos=batch, skip_predictor=True, return_dict=True)
                pooled = vjepa.pooler(encoded.last_hidden_state)
                logits = vjepa.classifier(pooled)
            pooled = functional.normalize(pooled.float(), dim=-1).cpu().numpy()
            logits = logits.float().cpu().numpy()
            block = np.concatenate((pooled[0], pooled[1], np.abs(pooled[0] - pooled[1]), logits[0], logits[1], np.abs(logits[0] - logits[1])), axis=0)
            vj_blocks.append(block.astype(np.float32))
    from PIL import Image
    center_ir = [Image.fromarray(ir_frames[t, FRAMES // 2]) for t in range(WINDOWS)]
    sig_batch = sig_processor(images=center_ir, return_tensors='pt')['pixel_values'].to('cuda', dtype=torch.float16)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        sig_features = siglip.get_image_features(pixel_values=sig_batch)
    if not isinstance(sig_features, torch.Tensor):
        sig_features = sig_features.pooler_output
    sig_features = functional.normalize(sig_features.float(), dim=-1).cpu().numpy()
    result = np.concatenate((np.stack(vj_blocks), sig_features, depth_pace, ir_pace), axis=1).astype(np.float32)
    if result.shape != (WINDOWS, 4374) or not np.isfinite(result).all():
        raise ExperimentError(f'joint feature shape drift: {result.shape}')
    return result

def make_head(input_dim: int, action_text: Any, emotion_text: Any, torch: Any):
    nn = torch.nn

    class TimelineHead(nn.Module):

        def __init__(self):
            super().__init__()
            hidden = 192
            self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15))
            self.position = nn.Parameter(torch.zeros(1, WINDOWS, hidden))
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=384, dropout=0.15, activation='gelu', batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.action_semantic = nn.Linear(hidden, 768)
            self.action_free = nn.Linear(hidden, 40)
            self.emotion_semantic = nn.Linear(hidden * 3, 768)
            self.action_scale = nn.Parameter(torch.tensor(math.log(12.0)))
            self.emotion_scale = nn.Parameter(torch.tensor(math.log(12.0)))
            self.register_buffer('action_text', action_text)
            self.register_buffer('emotion_text', emotion_text)
            nn.init.normal_(self.position, std=0.01)

        def forward(self, value):
            import torch.nn.functional as f
            hidden = self.temporal(self.input(value) + self.position)
            semantic = f.normalize(self.action_semantic(hidden), dim=-1)
            window = self.action_free(hidden) + self.action_scale.clamp(0, math.log(100)).exp() * (semantic @ self.action_text.T)
            tau = 0.5
            presence = tau * torch.logsumexp(window / tau, dim=1) - tau * math.log(WINDOWS)
            presence = 0.75 * presence + 0.25 * window.mean(dim=1)
            positions = torch.linspace(0.0, 1.0, WINDOWS, device=value.device).view(1, WINDOWS, 1)
            occurrence = (torch.softmax(window / 0.35, dim=1) * positions).sum(dim=1)
            pooled = torch.cat((hidden.mean(1), hidden.std(1, unbiased=False), hidden.amax(1)), dim=1)
            emotion_query = f.normalize(self.emotion_semantic(pooled), dim=-1)
            emotion = self.emotion_scale.clamp(0, math.log(100)).exp() * (emotion_query @ self.emotion_text.T)
            return (presence, occurrence, emotion)
    return TimelineHead()

def stable_softmax(value: np.ndarray) -> np.ndarray:
    shifted = value.astype(np.float64) - value.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)

def decode_test(rows: list[dict[str, str]], paths: list[str], emotion_terms: list[str], presence: np.ndarray, occurrence: np.ndarray, emotion: np.ndarray) -> list[dict[str, Any]]:
    pindex = {path: i for i, path in enumerate(paths)}
    eindex = {value: i for i, value in enumerate(emotion_terms)}
    output = []
    for row in rows:
        i, cat = (pindex[row['unit']], row['category'])
        if cat == 'single':
            actions = [parse_actions(row[letter])[0] for letter in LETTERS]
            prob = stable_softmax(presence[i, actions][None])[0]
            order = np.argsort(-prob)
            pred = LETTERS[int(order[0])]
            confidence, margin = (float(prob[order[0]]), float(prob[order[0]] - prob[order[1]]))
        elif cat == 'multi':
            actions = [parse_actions(row[letter])[0] for letter in LETTERS]
            prob = 1.0 / (1.0 + np.exp(-np.clip(presence[i, actions], -30, 30)))
            chosen = [letter for letter, value in zip(LETTERS, prob) if value >= 0.5]
            if not chosen:
                chosen = [LETTERS[int(np.argmax(prob))]]
            pred = ''.join(chosen)
            confidence = float(np.mean(np.maximum(prob, 1.0 - prob)))
            margin = float(np.min(np.abs(prob - 0.5)) * 2.0)
        elif cat == 'combination':
            masks = np.zeros((4, 40), dtype=np.float32)
            for j, letter in enumerate(LETTERS):
                masks[j, parse_actions(row[letter])] = 1.0
            union = masks.max(axis=0, keepdims=True)
            logp = -np.logaddexp(0, -presence[i])
            logn = -np.logaddexp(0, presence[i])
            positive = (masks * logp[None]).sum(1) / np.maximum(masks.sum(1), 1)
            negative_mask = np.maximum(union - masks, 0)
            negative = (negative_mask * logn[None]).sum(1) / np.maximum(negative_mask.sum(1), 1)
            prob = stable_softmax((positive + 0.5 * negative)[None])[0]
            order = np.argsort(-prob)
            pred = LETTERS[int(order[0])]
            confidence, margin = (float(prob[order[0]]), float(prob[order[0]] - prob[order[1]]))
        elif cat == 'sequence':
            actions = [parse_actions(row[letter])[0] for letter in LETTERS]
            times = occurrence[i, actions]
            order = np.argsort(times, kind='stable')
            pred = ''.join((LETTERS[int(j)] for j in order))
            gaps = np.diff(np.sort(times))
            confidence = float(np.clip(gaps.mean() * 4.0, 0, 1))
            margin = float(np.clip(gaps.min() * 4.0, 0, 1))
        elif cat == 'emotion':
            try:
                options = [eindex[row[letter]] for letter in LETTERS]
            except KeyError as error:
                raise ExperimentError(f'unknown anonymous emotion term: {error}') from error
            prob = stable_softmax(emotion[i, options][None])[0]
            order = np.argsort(-prob)
            pred = LETTERS[int(order[0])]
            confidence, margin = (float(prob[order[0]]), float(prob[order[0]] - prob[order[1]]))
        else:
            raise ExperimentError(cat)
        output.append({'qa_id': row['qa_id'], 'unit': row['unit'], 'category': cat, 'prediction': pred, 'confidence': confidence, 'margin': margin})
    if len(output) != len(rows) or len({row['qa_id'] for row in output}) != len(rows):
        raise ExperimentError('anonymous output contract failed')
    return output
