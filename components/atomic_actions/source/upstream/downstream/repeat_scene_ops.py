"""Recorded repeat-scene numerical operations; no top-level data reads."""
import math
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
N_CLASSES = 40
SCENE_GAP_SECONDS = 45.0
MAX_REPEAT_SPAN_SECONDS = 1500.0
SENSOR_TEMPERATURE = 1.25
SKELETON_TEMPERATURE = 1.25
SKELETON_WEIGHT = 0.3
EXACT_START_UNIQUENESS_WEIGHT = 0.15
REPEAT_MODES = (('ultra', 0.2), ('tight', 0.28), ('strict', 0.38), ('guarded', 0.52), ('balanced', 0.7), ('open', 0.92))
SCENE_WEIGHTS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
REPEAT_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)

def normalize_probabilities(values, temperature=1.0):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_CLASSES:
        raise ValueError(f'invalid probability shape {values.shape}')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('probabilities must be finite and nonnegative')
    values = np.maximum(values, 1e-12) ** (1.0 / float(temperature))
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError('zero probability row')
    values /= totals
    return values

def exact_start_batches(frame):
    work = frame[['start_time']].copy()
    work['_row'] = np.arange(len(work), dtype=np.int64)
    work['_start'] = pd.to_datetime(work['start_time'], errors='coerce')
    output = []
    for _, rows in work.dropna(subset=['_start']).groupby('_start', sort=True):
        output.append(rows['_row'].to_numpy(dtype=np.int64))
    covered = set(np.concatenate(output).tolist()) if output else set()
    for row in range(len(work)):
        if row not in covered:
            output.append(np.asarray([row], dtype=np.int64))
    if sorted(np.concatenate(output).tolist()) != list(range(len(work))):
        raise RuntimeError('exact-start batches do not partition rows')
    return output

def uniqueness_hard(probabilities, batches):
    probabilities = normalize_probabilities(probabilities)
    output = probabilities.copy()
    changed_batches = 0
    changed_rows = 0
    for rows in batches:
        if len(rows) <= 1:
            continue
        if len(rows) > N_CLASSES:
            raise RuntimeError('uniqueness group exceeds class count')
        row_indices, labels = linear_sum_assignment(-np.log(np.clip(probabilities[rows], 1e-12, 1.0)))
        hard = np.zeros((len(rows), N_CLASSES), dtype=np.float64)
        hard[row_indices, labels] = 1.0
        output[rows] = hard
        changed_batches += 1
        changed_rows += len(rows)
    return (output, changed_batches, changed_rows)

def frozen_baseline(sensor, skeleton, frame):
    sensor = normalize_probabilities(sensor, SENSOR_TEMPERATURE)
    skeleton = normalize_probabilities(skeleton, SKELETON_TEMPERATURE)
    fused = normalize_probabilities((1.0 - SKELETON_WEIGHT) * sensor + SKELETON_WEIGHT * skeleton)
    batches = exact_start_batches(frame)
    hard, batch_count, row_count = uniqueness_hard(fused, batches)
    final = normalize_probabilities((1.0 - EXACT_START_UNIQUENESS_WEIGHT) * fused + EXACT_START_UNIQUENESS_WEIGHT * hard)
    return (final, batch_count, row_count)

def build_scenes(frame, global_rows=None):
    work = frame[['start_time', 'end_time']].reset_index(drop=True).copy()
    if global_rows is None:
        global_rows = np.arange(len(work), dtype=np.int64)
    global_rows = np.asarray(global_rows, dtype=np.int64)
    if len(global_rows) != len(work) or len(np.unique(global_rows)) != len(global_rows):
        raise RuntimeError('invalid global row mapping')
    work['_global_row'] = global_rows
    work['_start'] = pd.to_datetime(work['start_time'], errors='coerce')
    work['_end'] = pd.to_datetime(work['end_time'], errors='coerce')
    valid = work.dropna(subset=['_start', '_end']).sort_values(['_start', '_end', '_global_row'], kind='mergesort')
    clusters = []
    current = []
    previous_start = None
    for local_row, row in valid.iterrows():
        start = row['_start']
        if previous_start is not None and float((start - previous_start).total_seconds()) > SCENE_GAP_SECONDS:
            clusters.append(current)
            current = []
        current.append(int(local_row))
        previous_start = start
    if current:
        clusters.append(current)
    covered = set((value for cluster in clusters for value in cluster))
    for local_row in range(len(work)):
        if local_row not in covered:
            clusters.append([local_row])
    scenes = []
    for local_rows in clusters:
        subset = work.iloc[local_rows].copy().sort_values(['_end', '_start', '_global_row'], kind='mergesort')
        rows = subset['_global_row'].to_numpy(dtype=np.int64)
        durations = (subset['_end'] - subset['_start']).dt.total_seconds().clip(lower=0.0).to_numpy(dtype=np.float64)
        start = subset['_start'].min()
        end = subset['_end'].max()
        order = np.arange(len(rows), dtype=np.float64)
        ranks = order / max(len(rows) - 1, 1)
        log_duration = np.log1p(np.maximum(durations, 0.0))
        centered = log_duration - np.median(log_duration)
        scale = max(float(np.median(np.abs(centered))), 0.35)
        scenes.append({'rows': rows, 'durations': durations, 'centered_log_duration': centered, 'duration_scale': scale, 'ranks': ranks, 'start': start, 'end': end})
    scenes.sort(key=lambda scene: (pd.Timestamp.max if pd.isna(scene['start']) else scene['start'], int(scene['rows'][0])))
    flattened = np.concatenate([scene['rows'] for scene in scenes])
    if sorted(flattened.tolist()) != sorted(global_rows.tolist()):
        raise RuntimeError('scenes do not partition target rows')
    for scene_index, scene in enumerate(scenes):
        scene['scene_index'] = int(scene_index)
        scene['uid'] = int(np.min(scene['rows']))
    return scenes

def scene_presence(probabilities, scene):
    values = probabilities[scene['rows']]
    presence = 1.0 - np.prod(1.0 - np.clip(values, 0.0, 1.0), axis=0)
    total = float(presence.sum())
    return presence / total if total > 0 else np.full(N_CLASSES, 1.0 / N_CLASSES)

def row_alignment_cost(probabilities, left, right):
    left_values = probabilities[left['rows']]
    right_values = probabilities[right['rows']]
    probability_distance = 1.0 - np.clip(np.sqrt(left_values) @ np.sqrt(right_values).T, 0.0, 1.0)
    duration = np.abs(left['centered_log_duration'][:, None] - right['centered_log_duration'][None, :]) / (left['duration_scale'] + right['duration_scale'])
    rank = np.abs(left['ranks'][:, None] - right['ranks'][None, :])
    geometry = np.minimum(0.7 * duration + 0.3 * rank, 2.0)
    return 0.52 * probability_distance + 0.48 * geometry

def scene_pair_cost(probabilities, left, right):
    alignment = row_alignment_cost(probabilities, left, right)
    row_indices, columns = linear_sum_assignment(alignment)
    aligned = float(alignment[row_indices, columns].mean())
    left_presence = scene_presence(probabilities, left)
    right_presence = scene_presence(probabilities, right)
    presence = 0.5 * float(np.abs(left_presence - right_presence).sum())
    left_size = len(left['rows'])
    right_size = len(right['rows'])
    count = abs(left_size - right_size) / max(left_size, right_size)
    return float(0.48 * aligned + 0.32 * presence + 0.55 * count)

def group_metrics(probabilities, scenes, indices):
    block = [scenes[index] for index in indices]
    pair_costs = [scene_pair_cost(probabilities, block[left], block[right]) for left in range(len(block)) for right in range(left + 1, len(block))]
    starts = [scene['start'] for scene in block]
    if any((pd.isna(value) for value in starts)):
        span = math.inf
    else:
        span = float((max(starts) - min(starts)).total_seconds())
    return {'mean_pair_cost': float(np.mean(pair_costs)) if pair_costs else math.nan, 'max_pair_cost': float(np.max(pair_costs)) if pair_costs else math.nan, 'span_seconds': span}

def infer_repeat_groups(probabilities, scenes, threshold):
    n = len(scenes)
    best = np.full(n + 1, np.inf, dtype=np.float64)
    trace = [None] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        singleton = best[end - 1] + 1.0
        best[end] = singleton
        trace[end] = (end - 1, (end - 1,), None)
        for size in (3, 2):
            start = end - size
            if start < 0:
                continue
            indices = tuple(range(start, end))
            metrics = group_metrics(probabilities, scenes, indices)
            if not np.isfinite(metrics['max_pair_cost']):
                continue
            if metrics['span_seconds'] > MAX_REPEAT_SPAN_SECONDS:
                continue
            if metrics['max_pair_cost'] > float(threshold):
                continue
            size_penalty = 0.0 if size == 3 else 0.22
            segment_cost = 0.1 + size_penalty + metrics['mean_pair_cost'] / max(float(threshold), 1e-09)
            candidate = best[start] + segment_cost
            if candidate < best[end] - 1e-12:
                best[end] = candidate
                trace[end] = (start, indices, metrics)
    groups = []
    cursor = n
    while cursor > 0:
        item = trace[cursor]
        if item is None:
            raise RuntimeError('repeat DP traceback failed')
        start, indices, metrics = item
        if metrics is None:
            metrics = {'mean_pair_cost': math.nan, 'max_pair_cost': math.nan, 'span_seconds': 0.0}
        groups.append({'scene_indices': tuple(indices), 'size': int(len(indices)), **metrics})
        cursor = int(start)
    groups.reverse()
    flattened = [index for group in groups for index in group['scene_indices']]
    if flattened != list(range(n)):
        raise RuntimeError('repeat groups do not partition scenes')
    return groups

def scene_uniqueness_hard(probabilities, scenes):
    return uniqueness_hard(probabilities, [scene['rows'] for scene in scenes])

def repeat_consensus_hard(probabilities, scenes, groups):
    probabilities = normalize_probabilities(probabilities)
    output = probabilities.copy()
    affected = np.zeros(len(probabilities), dtype=bool)
    records = []
    for group_index, group in enumerate(groups):
        if group['size'] <= 1:
            continue
        selected_scenes = [scenes[index] for index in group['scene_indices']]
        aggregated = []
        for target_index, target in enumerate(selected_scenes):
            sums = probabilities[target['rows']].copy()
            counts = np.ones(len(target['rows']), dtype=np.float64)
            for other_index, other in enumerate(selected_scenes):
                if other_index == target_index:
                    continue
                costs = row_alignment_cost(probabilities, target, other)
                target_rows, other_rows = linear_sum_assignment(costs)
                sums[target_rows] += probabilities[other['rows']][other_rows]
                counts[target_rows] += 1.0
            aggregated.append(normalize_probabilities(sums / counts[:, None]))
        common_count = min((len(scene['rows']) for scene in selected_scenes))
        class_scores = np.zeros(N_CLASSES, dtype=np.float64)
        for values in aggregated:
            class_scores += np.log(np.clip(values.max(axis=0), 1e-12, 1.0))
        common_labels = np.argsort(-class_scores, kind='mergesort')[:common_count]
        assigned_rows = 0
        for scene, values in zip(selected_scenes, aggregated):
            local_rows, local_columns = linear_sum_assignment(-np.log(np.clip(values[:, common_labels], 1e-12, 1.0)))
            hard = output[scene['rows']].copy()
            hard[local_rows] = 0.0
            hard[local_rows, common_labels[local_columns]] = 1.0
            output[scene['rows']] = hard
            affected[scene['rows']][local_rows] = True
            affected[scene['rows'][local_rows]] = True
            assigned_rows += len(local_rows)
        records.append({'group_index': int(group_index), 'size': int(group['size']), 'scene_indices': ','.join(map(str, group['scene_indices'])), 'scene_uids': ','.join((str(scenes[index]['uid']) for index in group['scene_indices'])), 'scene_row_counts': ','.join((str(len(scenes[index]['rows'])) for index in group['scene_indices'])), 'common_count': int(common_count), 'assigned_rows': int(assigned_rows), 'mean_pair_cost': float(group['mean_pair_cost']), 'max_pair_cost': float(group['max_pair_cost']), 'span_seconds': float(group['span_seconds'])})
    return (output, affected, records)

def candidate_specs():
    specs = [{'name': 'none', 'kind': 'none', 'complexity': 0.0}]
    for weight in SCENE_WEIGHTS:
        specs.append({'name': f'scene_w{weight:.2f}', 'kind': 'scene', 'scene_weight': float(weight), 'repeat_weight': 0.0, 'mode': None, 'complexity': float(weight)})
    for mode_index, (mode, _) in enumerate(REPEAT_MODES):
        for scene_weight in (0.0, 0.1, 0.2):
            for repeat_weight in REPEAT_WEIGHTS:
                specs.append({'name': f'repeat_{mode}_u{scene_weight:.2f}_r{repeat_weight:.2f}', 'kind': 'repeat', 'scene_weight': float(scene_weight), 'repeat_weight': float(repeat_weight), 'mode': mode, 'complexity': float(repeat_weight + scene_weight + 0.01 * mode_index)})
    return specs

def blend_spec(base, scene_hard, repeat_hard_by_mode, spec):
    if spec['kind'] == 'none':
        return base.copy()
    scene_weight = float(spec['scene_weight'])
    values = normalize_probabilities((1.0 - scene_weight) * base + scene_weight * scene_hard)
    if spec['kind'] == 'repeat':
        repeat_weight = float(spec['repeat_weight'])
        hard = repeat_hard_by_mode[spec['mode']]
        values = normalize_probabilities((1.0 - repeat_weight) * values + repeat_weight * hard)
    return values
