"""Recorded sensor sequence functions, extracted without notebook top-level code.

Only official observation metadata, training labels and freshly fitted model
probabilities are passed by sensor_stage. No filesystem reads occur here.
"""
import numpy as np
import pandas as pd
from numba import njit
N_CLASSES = 40
SOURCE_NOTEBOOK = "https://www.kaggle.com/code/ladyfaye/cuhk-x-small-sensor-dtw-v1"
SOURCE_SHA256 = "465bad5132cb064abe096f9c7977a3e80a93827430660462643a9cff04ec0ef8"

def build_sessions(frame, gap_seconds, by_user=True):
    work = frame.copy()
    work["_row"] = np.arange(len(work))
    work["_start"] = pd.to_datetime(work["start_time"], errors="coerce")
    work["_end"] = pd.to_datetime(work["end_time"], errors="coerce")
    partitions = work.groupby("user") if by_user else [("test", work)]
    sessions = []
    for partition, rows in partitions:
        rows = rows.dropna(subset=["_start"]).sort_values(["_start", "_end", "group"])
        current = []
        running_end = None
        for _, row in rows.iterrows():
            if running_end is not None and (row["_start"] - running_end).total_seconds() > gap_seconds:
                sessions.append({"partition": int(partition) if by_user else "test", "rows": np.asarray(current)})
                current = []
                running_end = None
            current.append(int(row["_row"]))
            running_end = row["_end"] if running_end is None else max(running_end, row["_end"])
        if current:
            sessions.append({"partition": int(partition) if by_user else "test", "rows": np.asarray(current)})
    return sessions


@njit(cache=True)
def align(costs, reference, hold_penalty, skip_penalty):
    target_length = costs.shape[0]
    reference_length = len(reference)
    infinity = 1e30
    dp = np.full((target_length + 1, reference_length + 1), infinity)
    move = np.zeros((target_length + 1, reference_length + 1), np.int8)
    dp[0, 0] = 0.0
    for reference_index in range(1, reference_length + 1):
        dp[0, reference_index] = dp[0, reference_index - 1] + skip_penalty
        move[0, reference_index] = 2
    for target_index in range(1, target_length + 1):
        for reference_index in range(1, reference_length + 1):
            emission = costs[target_index - 1, reference[reference_index - 1]]
            diagonal = dp[target_index - 1, reference_index - 1] + emission
            hold = dp[target_index - 1, reference_index] + emission + hold_penalty
            skip = dp[target_index, reference_index - 1] + skip_penalty
            if diagonal <= hold and diagonal <= skip:
                dp[target_index, reference_index] = diagonal
                move[target_index, reference_index] = 0
            elif hold <= skip:
                dp[target_index, reference_index] = hold
                move[target_index, reference_index] = 1
            else:
                dp[target_index, reference_index] = skip
                move[target_index, reference_index] = 2

    path = np.full(target_length, -1, np.int64)
    target_index = target_length
    reference_index = reference_length
    while target_index > 0 or reference_index > 0:
        selected = move[target_index, reference_index]
        if selected == 0:
            path[target_index - 1] = reference[reference_index - 1]
            target_index -= 1
            reference_index -= 1
        elif selected == 1:
            path[target_index - 1] = reference[reference_index - 1]
            target_index -= 1
        else:
            reference_index -= 1
    last = 0
    for index in range(target_length):
        if path[index] >= 0:
            last = path[index]
        else:
            path[index] = last
    return dp[target_length, reference_length] / max(target_length, 1), path


def sequence_probabilities(probabilities, labels, sessions, hold_penalty, skip_penalty, top_k, test=False):
    output = np.zeros_like(probabilities, dtype=np.float64)
    choices = []
    costs_all = -np.log(np.clip(probabilities, 1e-8, 1.0))
    references = [
        {"partition": session["partition"], "labels": labels[session["rows"]], "length": len(session["rows"])}
        for session in sessions["reference"]
    ]
    for target in sessions["target"]:
        target_rows = target["rows"]
        target_length = len(target_rows)
        candidates = []
        for reference_index, reference in enumerate(references):
            if not test and reference["partition"] == target["partition"]:
                continue
            ratio = reference["length"] / max(target_length, 1)
            if ratio < 0.42 or ratio > 2.40:
                continue
            score, path = align(
                costs_all[target_rows],
                reference["labels"].astype(np.int64),
                hold_penalty,
                skip_penalty,
            )
            candidates.append((float(score), reference_index, path))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        selected = candidates[: min(top_k, len(candidates))]
        values = np.asarray([item[0] for item in selected])
        weights = np.exp(-(values - values.min()) / 0.25)
        weights /= weights.sum()
        for weight, (_, _, path) in zip(weights, selected):
            output[target_rows] += weight * np.eye(N_CLASSES)[path]
        choices.append(
            {
                "target_partition": target["partition"],
                "target_length": target_length,
                "references": [
                    {"partition": references[index]["partition"], "length": references[index]["length"], "cost": score}
                    for score, index, _ in selected
                ],
            }
        )
    missing = output.sum(axis=1) == 0
    output[missing] = probabilities[missing]
    return output, choices
