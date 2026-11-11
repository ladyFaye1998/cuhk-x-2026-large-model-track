"""Compact Large Track structural expert.
Adapted by the original project from the Apache-2.0 code at
https://www.kaggle.com/code/phuongncn/lb-0-77777-exact-submission-research-handoff
"""
# Original recovered source SHA256: 34cfcc4a0dcb70605ba24b5a796c384f4e0a3f878d1918bf45ea1ba1d1ae2b1d
from __future__ import annotations
import itertools
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
LABELS = 'ABCD'
MODALITIES = {'depth', 'depth_color', 'ir', 'thermal', 'imu', 'skeleton', 'radar'}

def norm(value: str) -> str:
    return re.sub('\\s+', ' ', str(value).strip().lower())

def clip_key(path: str) -> str:
    parts = [piece for piece in str(path).replace('\\', '/').split('/') if piece]
    if len(parts) >= 2:
        modality, stem = (parts[-2].lower(), Path(parts[-1]).stem.lower())
        if modality in MODALITIES and stem in MODALITIES:
            parts = parts[:-2]
    return '/'.join(parts)

def atoms(value: str) -> tuple[str, ...]:
    return tuple((norm(piece) for piece in str(value).split(',') if piece.strip()))

class ClipGraphDecoder:
    """Readable public structural floor, kept faithful to the handoff."""

    def __init__(self, smoothing: float=8.0, support_weight: float=1.0):
        self.smoothing = smoothing
        self.support_weight = support_weight
        self.option_ok = Counter()
        self.option_n = Counter()
        self.slice_ok = Counter()
        self.slice_n = Counter()
        self.set_memory = defaultdict(Counter)
        self.multi_sizes = defaultdict(Counter)
        self.before = Counter()

    @staticmethod
    def options(row: pd.Series) -> list[str]:
        return [str(row[label]) for label in LABELS]

    def fit(self, frame: pd.DataFrame) -> 'ClipGraphDecoder':
        for _, row in frame.iterrows():
            answer = str(row['answer']).strip().upper()
            source, category = (row['source'], row['category'])
            option_values = self.options(row)
            option_set = tuple(sorted(map(norm, option_values)))
            slice_key = (source, category)
            if category != 'sequence':
                for label, value in zip(LABELS, option_values):
                    key = (source, category, norm(value))
                    self.option_n[key] += 1
                    self.slice_n[slice_key] += 1
                    if label in answer:
                        self.option_ok[key] += 1
                        self.slice_ok[slice_key] += 1
            if category == 'multi':
                self.multi_sizes[slice_key][len(answer)] += 1
            elif category == 'sequence' and set(answer) == set(LABELS):
                ordered = tuple((norm(row[label]) for label in answer))
                for index, left in enumerate(ordered):
                    for right in ordered[index + 1:]:
                        self.before[left, right] += 1
            else:
                for label in answer:
                    if label in LABELS:
                        self.set_memory[source, category, option_set][norm(row[label])] += 1
        return self

    def _logit(self, row: pd.Series, value: str) -> float:
        slice_key = (row['source'], row['category'])
        base = self.slice_ok[slice_key] / max(1, self.slice_n[slice_key])
        key = (row['source'], row['category'], norm(value))
        probability = (self.option_ok[key] + self.smoothing * base) / (self.option_n[key] + self.smoothing)
        probability = min(max(probability, 1e-06), 1 - 1e-06)
        return math.log(probability / (1 - probability))

    def _sequence(self, row: pd.Series) -> str:
        best = None
        for labels in itertools.permutations(LABELS):
            text = [norm(row[label]) for label in labels]
            score = 0.0
            for index, left in enumerate(text):
                for right in text[index + 1:]:
                    forward = self.before[left, right]
                    reverse = self.before[right, left]
                    score += math.log((forward + 1.5) / (forward + reverse + 3.0))
            candidate = (score, ''.join(labels))
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        return best[1]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        grouped = defaultdict(list)
        for index, row in frame.iterrows():
            grouped[clip_key(row['path'])].append((index, row))
        output = {}
        for group in grouped.values():
            support = Counter()
            for _, row in group:
                if row['source'] == 'HAU' and row['category'] in {'single', 'multi', 'combination', 'sequence'}:
                    values = self.options(row)
                    if row['category'] == 'combination':
                        for value in values:
                            support.update(atoms(value))
                    else:
                        support.update(map(norm, values))
            for _, row in group:
                category = row['category']
                if category == 'sequence':
                    output[row['qa_id']] = self._sequence(row)
                    continue
                values = self.options(row)
                option_set = tuple(sorted(map(norm, values)))
                memory = self.set_memory[row['source'], category, option_set]
                scores = []
                for value in values:
                    own = Counter(atoms(value) if category == 'combination' else [norm(value)])
                    external = sum((max(0, support[action] - own[action]) for action in own))
                    scores.append(self._logit(row, value) + math.log1p(memory[norm(value)]) + self.support_weight * external)
                if category == 'multi':
                    size_counts = self.multi_sizes[row['source'], category]
                    size = max(size_counts, key=lambda number: (size_counts[number], -number)) if size_counts else 1
                    selected = sorted(range(4), key=lambda idx: (-scores[idx], idx))[:size]
                    prediction = ''.join((LABELS[idx] for idx in sorted(selected)))
                else:
                    prediction = LABELS[max(range(4), key=lambda idx: (scores[idx], -idx))]
                output[row['qa_id']] = prediction
        return pd.DataFrame({'qa_id': frame['qa_id'], 'compact_prediction': frame['qa_id'].map(output)})
