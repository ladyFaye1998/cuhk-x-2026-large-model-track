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
FOLDS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (16, 17, 18), (19, 20, 21), (22, 23, 24))


def load_meta(path: Path) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    users, starts = [], []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        users.append(int(row["user"]))
        starts.append(row["start_time"])
    users_array = np.asarray(users, dtype=np.int64)
    buckets: dict[tuple[int, str], list[int]] = {}
    for index, key in enumerate(zip(users, starts)):
        buckets.setdefault(key, []).append(index)
    batches = [np.asarray(buckets[key], dtype=np.int64) for key in sorted(buckets)]
    fold_ids = np.full(len(rows), -1, dtype=np.int8)
    for fold, held in enumerate(FOLDS):
        fold_ids[np.isin(users_array, np.asarray(held, dtype=np.int64))] = fold
    if not np.array_equal(np.unique(fold_ids), np.arange(6, dtype=np.int8)):
        raise RuntimeError("fold coverage drift")
    return users_array, batches, fold_ids

def rank_matrix(probability: np.ndarray) -> np.ndarray:
    order = np.argsort(-probability, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(40, dtype=np.int64)[None, :]
    return ranks

def build_pair_table(
    views: list[np.ndarray],
    labels: np.ndarray,
    parent: np.ndarray,
    batches: list[np.ndarray],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, int]]:
    n_rows = len(labels)
    view_stack = np.stack(views, axis=1).astype(np.float64, copy=False)
    ranks = np.stack([rank_matrix(view) for view in views], axis=1)
    row_batch = np.empty(n_rows, dtype=np.int64)
    legal_unused: list[set[int]] = []
    batch_size = np.empty(n_rows, dtype=np.float64)
    for batch_index, rows in enumerate(batches):
        row_batch[rows] = batch_index
        used = set(parent[rows].tolist())
        legal_unused.append(set(range(40)) - used)
        batch_size[rows] = len(rows)
    pair_rows: list[int] = []
    pair_labels: list[int] = []
    per_row_candidates = np.zeros(n_rows, dtype=np.int64)
    top = np.argsort(-view_stack, axis=2, kind="stable")[:, :, :top_k]
    for row in range(n_rows):
        candidates = sorted(set(top[row].reshape(-1).tolist()))
        for challenger in candidates:
            if challenger == int(parent[row]) or challenger not in legal_unused[int(row_batch[row])]:
                continue
            pair_rows.append(row)
            pair_labels.append(int(challenger))
            per_row_candidates[row] += 1
    rows_array = np.asarray(pair_rows, dtype=np.int64)
    candidates_array = np.asarray(pair_labels, dtype=np.int64)
    if len(rows_array) == 0:
        raise RuntimeError("empty legal challenger table")
    row_axis = rows_array[:, None]
    view_axis = np.arange(len(views), dtype=np.int64)[None, :]
    challenger_axis = candidates_array[:, None]
    parent_axis = parent[rows_array][:, None]
    challenger_probability = view_stack[row_axis, view_axis, challenger_axis]
    parent_probability = view_stack[row_axis, view_axis, parent_axis]
    log_ratio = np.log(np.clip(challenger_probability, 1e-12, 1.0) / np.clip(parent_probability, 1e-12, 1.0))
    challenger_rank = ranks[row_axis, view_axis, challenger_axis].astype(np.float64) / 39.0
    challenger_top1 = (challenger_rank == 0.0).astype(np.float64)
    aggregates = np.column_stack(
        [
            challenger_top1.sum(axis=1),
            (log_ratio > 0).sum(axis=1),
            log_ratio.mean(axis=1),
            log_ratio.max(axis=1),
            log_ratio.min(axis=1),
            log_ratio.std(axis=1),
            batch_size[rows_array],
        ]
    )
    numeric = np.column_stack(
        [
            log_ratio,
            challenger_probability,
            parent_probability,
            challenger_rank,
            challenger_top1,
            aggregates,
        ]
    )
    categorical = np.zeros((len(rows_array), 80), dtype=np.float64)
    categorical[np.arange(len(rows_array)), parent[rows_array]] = 1.0
    categorical[np.arange(len(rows_array)), 40 + candidates_array] = 1.0
    features = np.column_stack([numeric, categorical])
    target = ((parent[rows_array] != labels[rows_array]) & (candidates_array == labels[rows_array])).astype(np.int8)
    diagnostics = {
        "pairs": len(rows_array),
        "rows_with_candidates": int(np.sum(per_row_candidates > 0)),
        "maximum_candidates_per_row": int(per_row_candidates.max()),
        "positive_pairs": int(target.sum()),
        "negative_pairs": int(len(target) - target.sum()),
    }
    return features, target, rows_array, candidates_array, numeric.shape[1], diagnostics

def logistic_model(c_value: float, numeric_columns: int, total_columns: int) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(range(numeric_columns))),
            ("categorical", "passthrough", list(range(numeric_columns, total_columns))),
        ],
        sparse_threshold=0.0,
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=20260823,
    )
    return Pipeline([("features", transformer), ("classifier", classifier)])

def hist_model(learning_rate: float, max_leaf_nodes: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=200,
        l2_regularization=1.0,
        min_samples_leaf=30,
        early_stopping=False,
        random_state=20260823,
    )

def crossfit_scores(
    features: np.ndarray,
    target: np.ndarray,
    pair_rows: np.ndarray,
    fold_ids: np.ndarray,
    model_family: str,
    parameters: dict[str, Any],
    numeric_columns: int,
) -> np.ndarray:
    scores = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(6):
        held = fold_ids[pair_rows] == fold
        fit = ~held
        if not np.any(held) or len(np.unique(target[fit])) != 2:
            raise RuntimeError(f"crossfit class or coverage drift fold{fold}")
        if model_family == "logistic":
            model = logistic_model(float(parameters["C"]), numeric_columns, features.shape[1])
            model.fit(features[fit], target[fit])
        elif model_family == "hist":
            model = hist_model(float(parameters["learning_rate"]), int(parameters["max_leaf_nodes"]))
            positive = int(target[fit].sum())
            negative = int(fit.sum() - positive)
            if positive == 0 or negative == 0:
                raise RuntimeError("balanced weight population drift")
            weights = np.where(target[fit] == 1, fit.sum() / (2.0 * positive), fit.sum() / (2.0 * negative))
            model.fit(features[fit], target[fit], sample_weight=weights)
        else:
            raise ValueError(model_family)
        scores[held] = model.predict_proba(features[held])[:, 1]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("crossfit score audit")
    return scores

def decode(
    parent: np.ndarray,
    pair_rows: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    batches: list[np.ndarray],
    threshold: float,
    minimum_gap: float,
    maximum_batch_changes: int,
) -> np.ndarray:
    output = parent.copy()
    by_row: dict[int, list[int]] = {}
    for pair_index, row in enumerate(pair_rows.tolist()):
        by_row.setdefault(row, []).append(pair_index)
    proposals: dict[int, tuple[float, float, int]] = {}
    for row, pair_indices in by_row.items():
        ordered = sorted(pair_indices, key=lambda index: (-scores[index], int(candidates[index])))
        best = ordered[0]
        second = float(scores[ordered[1]]) if len(ordered) > 1 else 0.0
        proposals[row] = (float(scores[best]), float(scores[best] - second), int(candidates[best]))
    for rows in batches:
        counts = Counter(output[rows].tolist())
        ordered_rows = sorted(
            (int(row) for row in rows if int(row) in proposals),
            key=lambda row: (-proposals[row][0], -proposals[row][1], row),
        )
        changes = 0
        for row in ordered_rows:
            score, gap, challenger = proposals[row]
            if changes >= maximum_batch_changes or score < threshold or gap < minimum_gap:
                continue
            if counts[challenger] != 0:
                continue
            old = int(output[row])
            counts[old] -= 1
            output[row] = challenger
            counts[challenger] += 1
            changes += 1
    return output
