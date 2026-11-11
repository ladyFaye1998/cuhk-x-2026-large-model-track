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


def normalize_logits(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all():
        raise RuntimeError("non-finite probability")
    return probability

def collision_count(prediction: np.ndarray, batches: list[np.ndarray]) -> int:
    return int(sum(len(rows) - len(np.unique(prediction[rows])) for rows in batches))

def ranked_decode(
    parent: np.ndarray,
    rows: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    batches: list[np.ndarray],
    topk: int,
) -> tuple[np.ndarray, list[float]]:
    prediction = np.asarray(parent, dtype=np.int64).copy()
    row_to_batch = np.full(len(prediction), -1, dtype=np.int64)
    for batch_id, batch_rows in enumerate(batches):
        row_to_batch[np.asarray(batch_rows, dtype=np.int64)] = batch_id
    if np.any(row_to_batch < 0):
        raise RuntimeError("incomplete batch coverage")
    parent_collisions = collision_count(prediction, batches)
    changed_rows: set[int] = set()
    changed_batches: set[int] = set()
    accepted_scores: list[float] = []
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    for pair_index in order.tolist():
        row = int(rows[pair_index])
        candidate = int(candidates[pair_index])
        batch_id = int(row_to_batch[row])
        if row in changed_rows or batch_id in changed_batches or candidate == int(parent[row]):
            continue
        proposed = prediction.copy()
        proposed[row] = candidate
        if collision_count(proposed, batches) > parent_collisions:
            continue
        prediction = proposed
        changed_rows.add(row)
        changed_batches.add(batch_id)
        accepted_scores.append(float(scores[pair_index]))
        if len(changed_rows) == topk:
            break
    if len(changed_rows) != topk:
        raise RuntimeError(f"only {len(changed_rows)} of {topk} ranked interventions were feasible")
    return prediction, accepted_scores
