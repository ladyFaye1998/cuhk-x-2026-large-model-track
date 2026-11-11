"""Train the six recorded joint/bone SkateFormer teachers from observations.

Public NTU120 initialization is the only learned input. Training uses exactly the
recorded 24-epoch recipe; no held-out or test result selects the epoch or settings.
This executor saves FP32 states and the original FP16 replay for every fold.
The portable inference stage consumes checkpoint_map.json after training.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

# Must be set before CUDA BLAS is initialized when deterministic algorithms run.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import skateformer_training_vendor as recipe

ROOT = Path(__file__).resolve().parent
FOLDS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (16, 17, 18), (19, 20, 21), (22, 23, 24))
HELD_COUNTS = (477, 435, 530, 530, 478, 481)
PUBLIC_HASHES = {'joint': 'ac3b76a995cd9faa45bf184a1625c3730f590bd41f62acc340bc8105fc36ede3',
                 'bone': 'deb19c175afd9e0cdbb14bf47ff8b998073b775f603d3533babc5bcb0a0b966a'}

def fact(path, role):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': recipe.sha256_file(path), 'bytes': path.stat().st_size, 'role': role}

def load_module(name):
    path = ROOT / 'vendor' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def load_observations(cache, metadata_path):
    # Test tensors are not materialized during fitting.
    with np.load(cache, allow_pickle=False) as archive:
        if set(archive.files) not in ({'train', 'train_n_valid'}, {'train', 'train_n_valid', 'test', 'test_n_valid'}):
            raise ValueError('Only observation arrays are accepted')
        train = np.asarray(archive['train'], dtype=np.float32)
        lengths = np.asarray(archive['train_n_valid'], dtype=np.int64)
    with Path(metadata_path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if train.shape != (2931, 128, 17, 3) or lengths.shape != (2931,) or len(rows) != 2931:
        raise ValueError('Official training observation shape mismatch')
    labels = np.asarray([int(r['label']) for r in rows], np.int64)
    users = np.asarray([int(r['user'].removeprefix('user')) for r in rows], np.int64)
    groups = tuple(r['group'] for r in rows)
    if len(set(groups)) != 2931 or not np.isfinite(train).all():
        raise ValueError('Invalid observations or duplicate row identities')
    if not np.array_equal(lengths, np.asarray([int(r['n_valid']) for r in rows], np.int64)):
        raise ValueError('Metadata/observation valid lengths differ')
    if np.any((lengths < 1) | (lengths > 128)) or np.any((labels < 0) | (labels >= 40)):
        raise ValueError('Invalid labels or observation lengths')
    for held, count in zip(FOLDS, HELD_COUNTS):
        if np.isin(users, held).sum() != count:
            raise ValueError('Canonical subject partition mismatch')
    return train, recipe.Metadata(labels, users, lengths, groups)

def build_checkpoint_map(checkpoints, output_root, folds):
    """Write the inference stage's fold schema, with portable relative paths.

    Partial training jobs deliberately produce a partial map; the inference
    stage requires six complete folds and therefore rejects such maps.
    """
    root = Path(output_root).resolve()
    expected = {(fold, modality) for fold in folds for modality in ('joint', 'bone')}
    by_key = {(item['fold'], item['modality']): item for item in checkpoints}
    if len(by_key) != len(checkpoints) or set(by_key) != expected:
        raise ValueError('Checkpoint map requires exactly one complete pair per trained fold')
    records = []
    for fold in sorted(folds):
        record = {'fold': fold, 'held_users': list(FOLDS[fold])}
        for modality in ('joint', 'bone'):
            item = by_key[(fold, modality)]
            path = Path(item['path']).resolve()
            record[modality] = {
                'path': path.relative_to(root).as_posix(),
                'sha256': item['sha256'],
                'canonical_state_sha256': item['canonical_state_sha256'],
            }
        records.append(record)
    complete = sorted(folds) == list(range(6))
    return {'schema': 'skateformer-fold-checkpoint-map-v1',
            'status': 'COMPLETE' if complete else 'PARTIAL_NOT_READY_FOR_SIX_FOLD_INFERENCE',
            'ready_for_six_fold_inference': complete, 'folds_completed': sorted(folds),
            'folds': records,
            'fold0_mapping': 'Fresh backbone-transfer treatment from the fixed original training recipe; semantic-head treatment is not substituted.'}

def run_fit(skeleton_cache, train_metadata, public_root, output_root, folds=tuple(range(6))):
    # CUDA is required unconditionally. The Linux requirement is relaxed only on a
    # declared host (CUHKX_ALLOW_NON_LINUX_HOST=1); the deviation is recorded in the run
    # report so a reviewer can see fitting did not run on the original worker image.
    if not torch.cuda.is_available():
        raise RuntimeError('Full fitting requires CUDA')
    if platform.system() != 'Linux' and os.environ.get('CUHKX_ALLOW_NON_LINUX_HOST') != '1':
        raise RuntimeError('Full fitting requires the original Linux/CUDA worker environment')
    if not folds or len(set(folds)) != len(folds) or any(f not in range(6) for f in folds):
        raise ValueError('Specify unique canonical folds 0..5')
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Training output must be empty; do not silently reuse states')
    started = time.time()
    inputs = [fact(skeleton_cache, 'official_training_skeleton_observations'), fact(train_metadata, 'official_training_row_metadata')]
    public = {}
    for modality, expected in PUBLIC_HASHES.items():
        path = Path(public_root) / f'ntu120_xsub_{modality}.safetensors'
        entry = fact(path, 'public_NTU120_pretraining_initialization')
        if entry['sha256'] != expected:
            raise ValueError('Public initialization hash mismatch: '+modality)
        public[modality] = path
        inputs.append(entry)
    model_module = load_module('SkateFormer_official')
    adapter = load_module('skateformer_locked_preprocessing_v1')
    official_tools = load_module('feeder_tools_official')
    train, metadata = load_observations(skeleton_cache, train_metadata)
    raw = recipe.prepare_raw_ntu(train, metadata, adapter)
    del train
    source = {name: recipe.load_source_state(path) for name, path in public.items()}
    for state in source.values():
        recipe.validate_source_state(model_module, state)
    initial = {}
    loading = {}
    for modality in ('joint', 'bone'):
        control, treatment, load_receipt = recipe.build_initial_states(model_module, source[modality], modality)
        initial[modality] = treatment
        loading[modality] = load_receipt
        del control
    del source
    device = torch.device('cuda:0')
    checkpoints = []
    reports = []
    for fold in folds:
        recipe.FOLD = fold
        recipe.HELD_USERS = FOLDS[fold]
        recipe.EXPECTED_HELD_ROWS = HELD_COUNTS[fold]
        fit_rows = np.flatnonzero(~np.isin(metadata.users, FOLDS[fold]))
        recipe.EXPECTED_BATCHES_PER_EPOCH = math.ceil(len(fit_rows) / recipe.BATCH_SIZE)
        recipe.EXPECTED_OPTIMIZER_STEPS_PER_ARM = recipe.EPOCHS * recipe.EXPECTED_BATCHES_PER_EPOCH
        fold_root = output / f'fold{fold}'
        fold_root.mkdir()
        for modality in ('joint', 'bone'):
            dataset = recipe.SkateDataset(raw, metadata, modality, True, adapter, official_tools, fold=fold)
            state, training = recipe.train_final_state('treatment_'+modality, modality, initial[modality], model_module, dataset, fit_rows, device)
            full_path = fold_root / (modality+'_fp32.pt')
            torch.save(state, full_path)
            replay_path = fold_root / (modality+'_fp16.safetensors')
            replay, replay_receipt = recipe.make_symmetric_float16_replay('treatment_'+modality, state, replay_path)
            checkpoint = {'fold': fold, 'modality': modality, 'path': str(replay_path),
                          'sha256': recipe.sha256_file(replay_path),
                          'canonical_state_sha256': recipe.canonical_tensor_digest(replay.items())}
            checkpoints.append(checkpoint)
            reports.append({'fold': fold, 'held_users': list(FOLDS[fold]), 'training': training,
                            'fp32_checkpoint': fact(full_path, 'fresh_trained_FP32_fold_state'),
                            'fp16_replay': replay_receipt, 'replay_checkpoint': checkpoint})
            (output / 'training_progress.json').write_text(json.dumps(reports, indent=2)+'\n')
            del state, replay, dataset
            gc.collect()
    mapping = build_checkpoint_map(checkpoints, output, folds)
    (output / 'checkpoint_map.json').write_text(json.dumps(mapping, indent=2)+'\n')
    sources = [fact(path, 'executable_source') for path in sorted(ROOT.rglob('*.py'))]
    manifest = {'stage': 'skateformer_fit', 'inputs': inputs, 'source_files': sources,
                'learned_inputs_used': [entry for entry in inputs if entry['role'] == 'public_NTU120_pretraining_initialization'],
                'folds_completed': list(folds), 'all_six_folds_trained': sorted(folds) == list(range(6)),
                'method': {'epochs': 24, 'batch': 32, 'control_arm_retrained': False,
                           'transfer_arm_training_unchanged': True, 'held_evaluations_during_training': 0,
                           'selection': 'Fixed historical training settings and final epoch; no candidate search.',
                           'omitted_control_reason': 'Original train_final_state resets RNG at the start of each arm; random-control states are not inputs to transfer fitting.'},
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'device': torch.cuda.get_device_name(0),
                            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                            'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                            'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32)},
                'partial_loads': loading, 'training_reports': reports, 'seconds': time.time()-started,
                'technical_scope': 'fresh fixed-recipe training and saved-state inference'}
    (output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skeleton-cache', required=True)
    parser.add_argument('--train-metadata', required=True)
    parser.add_argument('--public-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(6)))
    args = parser.parse_args()
    result = run_fit(args.skeleton_cache, args.train_metadata, args.public_root, args.output_root, tuple(args.folds))
    print(json.dumps({'folds_completed': result['folds_completed'], 'seconds': result['seconds']}))

if __name__ == '__main__':
    main()
