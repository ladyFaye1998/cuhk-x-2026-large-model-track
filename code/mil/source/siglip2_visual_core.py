"""Original raw-vision functions; experiment orchestration removed."""

from __future__ import annotations

# Source SHA256: 667d968c144f050cd2b7fb086346b7029fba87298310b1182a4fe9b98a553502

import re
import zipfile
import cv2
import numpy as np
from PIL import Image

N_FRAMES = 8

def training_unit(name):
    match = re.search(r"(?:^|/)(HAU/user\d+/[^/]+)/", name, re.I)
    if match:
        parts = match.group(1).split("/")
        parts[1] = parts[1].lower()
        return "/".join(parts)
    match = re.search(r"(?:^|/)(HARn/[^/]+/user\d+/[^/]+)/", name, re.I)
    if match:
        parts = match.group(1).split("/")
        parts[2] = parts[2].lower()
        return "/".join(parts)
    return None

def video_priority(name):
    low = name.lower().replace("\\", "/")
    if "/depth_color/" in low:
        return 0
    if "/ir/" in low:
        return 1
    if "/depth/" in low:
        return 2
    if "/thermal/" in low:
        return 3
    return 10

def archive_entries(archive_path):
    selected = {}
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                continue
            unit = training_unit(info.filename)
            if not unit:
                continue
            priority = video_priority(info.filename)
            if priority >= 10:
                continue
            if unit not in selected or priority < selected[unit][0]:
                selected[unit] = (priority, info.filename)
    return {unit: member for unit, (_, member) in selected.items()}

def sample_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        return []
    targets = np.linspace(0, max(0, count - 1), N_FRAMES).round().astype(int)
    frames = []
    for target in targets:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    capture.release()
    if frames and len(frames) < N_FRAMES:
        frames.extend([frames[-1].copy() for _ in range(N_FRAMES - len(frames))])
    return frames[:N_FRAMES]
