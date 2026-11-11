"""Apply the recorded repeat-scene and fusion recipe to regenerated teachers."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from assemble_teacher_inputs import FOLDS, read_stage, sha
import repeat_scene_ops as repeat

REPEAT_RECIPE = 'repeat_tight_u0.10_r0.30'
FUSION_WEIGHTS = {'structure': 0.85, 'adaptive': 0.15, 'adaptive_temperature': 0.75}


def normalize(values, temperature=1.0, logits=False):
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[1] != 40 or not np.isfinite(values).all():
        raise ValueError('Invalid prediction matrix')
    if logits:
        values = values / float(temperature)
        values -= values.max(axis=1, keepdims=True)
        values = np.exp(values)
    else:
        if np.min(values) < 0:
            raise ValueError('Negative probability')
        values = np.maximum(values, 1e-12) ** (1.0 / float(temperature))
    values /= values.sum(axis=1, keepdims=True)
    return values


def repeat_view(sensor, kinematic, metadata, users=None):
    base, _, _ = repeat.frozen_baseline(sensor.astype(np.float64), kinematic.astype(np.float64), metadata)
    result = base.copy()
    spec = next(s for s in repeat.candidate_specs() if s['name'] == REPEAT_RECIPE)
    threshold = dict(repeat.REPEAT_MODES)[spec['mode']]
    groups_of_rows = [np.arange(len(base))] if users is None else [np.flatnonzero(np.isin(users, held)) for held in FOLDS]
    for rows in groups_of_rows:
        scenes = repeat.build_scenes(metadata.iloc[rows], global_rows=rows)
        unique, _, _ = repeat.scene_uniqueness_hard(base, scenes)
        groups = repeat.infer_repeat_groups(base, scenes, threshold)
        consensus, _, _ = repeat.repeat_consensus_hard(base, scenes, groups)
        values = repeat.blend_spec(base, unique, {spec['mode']: consensus}, spec)
        result[rows] = values[rows]
    # The repeat producer serialized float32 before the fusion consumer.
    return normalize(result.astype(np.float32))


def run_fusion(sensor_dir, kinematic_dir, adaptive_dir, train_meta, test_meta, out):
    started = time.perf_counter()
    train, test = pd.read_csv(train_meta), pd.read_csv(test_meta)
    train_groups, test_groups = train.group.to_numpy(dtype=str), test.group.to_numpy(dtype=str)
    labels, users = train.label.to_numpy(np.int64), train.user.to_numpy(np.int64)
    values, receipts = {}, []
    for stage, path in (('sensor', sensor_dir), ('kinematic', kinematic_dir), ('raw_adaptive', adaptive_dir)):
        arrays, receipt = read_stage(stage, path, train_groups, test_groups, labels, users)
        values.update(arrays)
        receipts.append(receipt)
    result = {}
    for split, metadata in (('train', train), ('test', test)):
        structure = repeat_view(values[split+'_sensor'], values[split+'_kinematic'], metadata,
                                users if split == 'train' else None)
        adaptive = normalize(values[split+'_raw_logits'], FUSION_WEIGHTS['adaptive_temperature'], logits=True)
        result[split+'_fusion'] = normalize(FUSION_WEIGHTS['structure']*structure +
                                            FUSION_WEIGHTS['adaptive']*adaptive).astype(np.float32)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'teacher_arrays.npz'
    if path.exists():
        raise FileExistsError('Use a new stage output directory')
    np.savez_compressed(path, **result, labels=labels, users=users,
                        train_groups=train_groups, test_groups=test_groups)
    inputs = [{'path': str(Path(p).resolve()), 'sha256': sha(p), 'role': role}
              for p, role in ((train_meta, 'official_train_metadata'), (test_meta, 'official_test_metadata'))]
    for stage, directory in (('sensor', sensor_dir), ('kinematic', kinematic_dir), ('raw_adaptive', adaptive_dir)):
        source = Path(directory) / 'teacher_arrays.npz'
        inputs.append({'path': str(source.resolve()), 'sha256': sha(source), 'role': 'regenerated_'+stage})
    source_files = [Path(__file__), Path(repeat.__file__), Path(__file__).with_name('assemble_teacher_inputs.py')]
    manifest = {'stage': 'fusion', 'array_file': path.name, 'array_sha256': sha(path),
                'inputs': inputs, 'source_files': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in source_files],
                'learned_inputs_used': inputs[2:], 'stage_receipts': receipts,
                'method': {'repeat_spec': REPEAT_RECIPE, 'fusion_weights': FUSION_WEIGHTS,
                           'selection': 'Fixed recorded recipe, no new parameter search or reference comparison.'},
                'seconds': time.perf_counter()-started,
                'technical_scope': 'Fixed arithmetic assembly of regenerated upstream stage arrays.'}
    (out / 'stage_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sensor-dir', 'kinematic-dir', 'adaptive-dir', 'train-meta', 'test-meta', 'out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(run_fusion(args.sensor_dir, args.kinematic_dir, args.adaptive_dir,
                               args.train_meta, args.test_meta, args.out), indent=2))
