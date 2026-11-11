from __future__ import annotations
import csv
from pathlib import Path
from collections import Counter
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
N_CLASSES=40


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    values /= values.sum(1, keepdims=True)
    if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1.0, atol=1e-9):
        raise RuntimeError("invalid probabilities")
    return values

def batch_postprocess(probabilities: np.ndarray, batches: list[np.ndarray]) -> np.ndarray:
    output = probabilities.copy()
    for rows in batches:
        if len(rows) <= 1:
            continue
        plain = probabilities[rows].argmax(1)
        if len(np.unique(plain)) == len(rows):
            continue
        rr, cc = linear_sum_assignment(-np.log(np.clip(probabilities[rows], 1e-12, 1.0)))
        assignment = np.empty(len(rows), dtype=np.int64)
        assignment[rr] = cc
        if int(np.sum(assignment != plain)) > 2:
            continue
        one_hot = np.zeros((len(rows), N_CLASSES), dtype=np.float64)
        one_hot[np.arange(len(rows)), assignment] = 1.0
        output[rows] = normalize(0.25 * probabilities[rows] + 0.75 * one_hot)
    return normalize(output)

def consensus(teachers: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.stack([values.argmax(1) for values in teachers], axis=1)
    labels = np.empty(len(predictions), dtype=np.int64)
    support = np.empty(len(predictions), dtype=np.int64)
    confidence = np.empty(len(predictions), dtype=np.float64)
    for row, votes in enumerate(predictions):
        counts = np.bincount(votes, minlength=N_CLASSES)
        label = int(counts.argmax())
        labels[row] = label
        support[row] = int(counts[label])
        confidence[row] = float(np.mean([teacher[row, label] for teacher in teachers]))
    return labels, support, confidence

def correction(anchor: np.ndarray, consensus_label: np.ndarray, support: np.ndarray, teacher_confidence: np.ndarray, min_support: int, max_anchor: float, min_teacher: float) -> np.ndarray:
    output = anchor.copy()
    base = anchor.argmax(1)
    mask = (support >= min_support) & (consensus_label != base) & (anchor.max(1) <= max_anchor) & (teacher_confidence >= min_teacher)
    for row in np.flatnonzero(mask):
        left, right = int(base[row]), int(consensus_label[row])
        output[row, left], output[row, right] = output[row, right], output[row, left]
    return normalize(output)
