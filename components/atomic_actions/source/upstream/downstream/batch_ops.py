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


def align_probabilities(probability: np.ndarray, parent: np.ndarray) -> np.ndarray:
    output = probability.astype(np.float64, copy=True)
    plain = output.argmax(1)
    for row in np.flatnonzero(plain != parent):
        left, right = int(plain[row]), int(parent[row])
        output[row, left], output[row, right] = output[row, right], output[row, left]
    output = np.clip(output, 1e-12, None)
    output /= output.sum(1, keepdims=True)
    if not np.array_equal(output.argmax(1), parent):
        raise RuntimeError("parent probability alignment failed")
    return output

def batch_rows(frame: pd.DataFrame, train: bool) -> list[np.ndarray]:
    work = frame.assign(_row=np.arange(len(frame), dtype=np.int64))
    keys = ["user", "start_time"] if train else ["start_time"]
    return [part._row.to_numpy(np.int64) for _, part in work.groupby(keys, sort=True, dropna=False)]
