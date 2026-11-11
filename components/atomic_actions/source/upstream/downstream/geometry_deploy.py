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


def test_batches(path: Path) -> list[np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = [f"SM_test_{index:04d}" for index in range(1, 406)]
    if len(rows) != 405 or [row["group"] for row in rows] != expected:
        raise RuntimeError("test metadata row-order drift")
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault(row["start_time"], []).append(index)
    return [np.asarray(buckets[key], dtype=np.int64) for key in sorted(buckets)]

def fullfit_geometry_views(
    research: Any,
    train_features: np.ndarray,
    test_features: np.ndarray,
    labels: np.ndarray,
) -> list[np.ndarray]:
    train = research.normalize_rows(train_features)
    test = research.normalize_rows(test_features)
    centroids = np.empty((40, train.shape[1]), dtype=np.float64)
    for class_id in range(40):
        centroid = train[labels == class_id].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("invalid fullfit centroid")
        centroids[class_id] = centroid / norm
    centroid_similarity = test @ centroids.T
    centroid_views = [research.softmax(centroid_similarity, temperature) for temperature in research.CENTROID_TEMPERATURES]
    neighbor_similarity = test @ train.T
    knn_views = [research.knn_probabilities(neighbor_similarity, labels, k_value) for k_value in research.KNN_K]
    return centroid_views + knn_views
