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


def evidence(anchor: np.ndarray, teachers: list[np.ndarray], parent_prediction: np.ndarray) -> np.ndarray:
    rows = np.arange(len(anchor))
    anchor_prediction = anchor.argmax(1)
    teacher_stack = np.stack(teachers, axis=1)
    teacher_parent = teacher_stack[rows[:, None], np.arange(len(teachers))[None, :], parent_prediction[:, None]]
    teacher_anchor = teacher_stack[rows[:, None], np.arange(len(teachers))[None, :], anchor_prediction[:, None]]
    teacher_log_odds = np.log(np.clip(teacher_parent, 1e-12, 1.0) / np.clip(teacher_anchor, 1e-12, 1.0)).mean(1)
    anchor_log_odds = np.log(
        np.clip(anchor[rows, anchor_prediction], 1e-12, 1.0) / np.clip(anchor[rows, parent_prediction], 1e-12, 1.0)
    )
    return teacher_log_odds - anchor_log_odds

def prune(anchor: np.ndarray, teachers: list[np.ndarray], parent_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_prediction = anchor.argmax(1)
    changed = np.flatnonzero(parent_prediction != anchor_prediction)
    if len(changed) == 0:
        raise RuntimeError("parent has no changes")
    count = max(1, round(len(changed) * 0.20))
    reliability = evidence(anchor, teachers, parent_prediction)
    ordered = changed[np.argsort(reliability[changed], kind="stable")]
    reverted = ordered[:count]
    output = parent_prediction.copy()
    output[reverted] = anchor_prediction[reverted]
    return output, reverted, reliability
