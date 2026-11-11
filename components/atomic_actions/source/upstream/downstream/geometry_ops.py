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
CENTROID_TEMPERATURES=(.025,.05,.1)
KNN_K=(1,3,7,15,31)
KNN_TEMPERATURE=.05


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise RuntimeError("invalid feature rows")
    return values / norms

def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    probability /= probability.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1.0, atol=1e-9):
        raise RuntimeError("invalid softmax probability")
    return probability

def knn_probabilities(similarity: np.ndarray, fit_labels: np.ndarray, k: int) -> np.ndarray:
    if similarity.shape[1] < k:
        raise RuntimeError("insufficient kNN fit rows")
    indices = np.argpartition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
    scores = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    indices = np.take_along_axis(indices, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    weights = np.exp((scores - scores[:, :1]) / KNN_TEMPERATURE)
    result = np.full((len(similarity), 40), 1e-6, dtype=np.float64)
    neighbor_labels = fit_labels[indices]
    for column in range(k):
        result[np.arange(len(result)), neighbor_labels[:, column]] += weights[:, column]
    result /= result.sum(axis=1, keepdims=True)
    return result

def cross_subject_geometry_views(
    features: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, float]]:
    normalized = normalize_rows(features)
    centroid_views = [np.zeros((len(labels), 40), dtype=np.float64) for _ in CENTROID_TEMPERATURES]
    knn_views = [np.zeros((len(labels), 40), dtype=np.float64) for _ in KNN_K]
    for fold in range(6):
        held = np.flatnonzero(fold_ids == fold)
        fit = np.flatnonzero(fold_ids != fold)
        centroids = np.empty((40, normalized.shape[1]), dtype=np.float64)
        for class_id in range(40):
            class_rows = fit[labels[fit] == class_id]
            if len(class_rows) == 0:
                raise RuntimeError(f"missing class {class_id} in fold {fold} fit")
            centroid = normalized[class_rows].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if not np.isfinite(norm) or norm <= 1e-12:
                raise RuntimeError("invalid class centroid")
            centroids[class_id] = centroid / norm
        centroid_similarity = normalized[held] @ centroids.T
        for index, temperature in enumerate(CENTROID_TEMPERATURES):
            centroid_views[index][held] = softmax(centroid_similarity, temperature)
        neighbor_similarity = normalized[held] @ normalized[fit].T
        for index, k_value in enumerate(KNN_K):
            knn_views[index][held] = knn_probabilities(neighbor_similarity, labels[fit], k_value)
    views = centroid_views + knn_views
    diagnostics = {
        f"centroid_t{temperature:g}_accuracy": float(np.mean(view.argmax(1) == labels))
        for temperature, view in zip(CENTROID_TEMPERATURES, centroid_views)
    }
    diagnostics.update(
        {
            f"knn_k{k_value}_accuracy": float(np.mean(view.argmax(1) == labels))
            for k_value, view in zip(KNN_K, knn_views)
        }
    )
    return views, diagnostics
