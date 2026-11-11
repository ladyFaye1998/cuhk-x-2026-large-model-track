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


def batch_design(views: list[np.ndarray], batches: list[np.ndarray]) -> np.ndarray:
    features = np.empty((len(batches), N_CLASSES, len(views) * 3 + 2 + N_CLASSES), dtype=np.float64)
    eye = np.eye(N_CLASSES, dtype=np.float64)
    for batch_index, rows in enumerate(batches):
        columns = []
        for view in views:
            values = view[rows]
            columns.extend([values.max(0), values.mean(0), np.quantile(values, 0.75, axis=0)])
        size = np.full(N_CLASSES, len(rows) / 10.0, dtype=np.float64)
        entropy = np.zeros(N_CLASSES, dtype=np.float64)
        mean_anchor = views[0][rows].mean(0)
        entropy[:] = -float(np.sum(mean_anchor * np.log(np.clip(mean_anchor, 1e-12, 1.0)))) / np.log(N_CLASSES)
        numeric = np.column_stack([*columns, size, entropy])
        features[batch_index] = np.column_stack([numeric, eye])
    return features

def batch_targets(labels: np.ndarray, batches: list[np.ndarray]) -> np.ndarray:
    target = np.zeros((len(batches), N_CLASSES), dtype=np.int8)
    for batch_index, rows in enumerate(batches):
        target[batch_index, np.unique(labels[rows])] = 1
    return target

def decode(
    probability: np.ndarray,
    parent: np.ndarray,
    batches: list[np.ndarray],
    presence: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    output = parent.copy()
    reports = []
    for batch_index, rows in enumerate(batches):
        score = np.log(np.clip(probability[rows], 1e-12, 1.0)) + weight * np.log(np.clip(presence[batch_index], 1e-12, 1.0))[None, :]
        rr, cc = linear_sum_assignment(-score)
        assigned = np.empty(len(rows), dtype=np.int64)
        assigned[rr] = cc
        parent_local = parent[rows]
        gain = float(score[np.arange(len(rows)), assigned].sum() - score[np.arange(len(rows)), parent_local].sum())
        changes = int(np.sum(assigned != parent_local))
        output[rows] = assigned
        reports.append({"gain": gain, "changes": changes, "size": len(rows)})
    return output, reports

def gate(parent: np.ndarray, decoded: np.ndarray, batches: list[np.ndarray], reports: list[dict[str, float | int]], max_changes: int, min_gain: float) -> np.ndarray:
    output = parent.copy()
    for rows, report in zip(batches, reports):
        if 1 <= int(report["changes"]) <= max_changes and float(report["gain"]) >= min_gain:
            output[rows] = decoded[rows]
    return output
