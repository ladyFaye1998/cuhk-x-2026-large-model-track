"""Original SigLIP2/sensor MIL and graph model math."""
# Recovered source SHA256: edc6b53198e233888f6688248fcab02f2f75d19f28e3fd95092a24cdf1246894
from __future__ import annotations
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
INPUT_ROOT = Path('/kaggle/input')
LETTERS = 'ABCD'
PERMUTATIONS = list(itertools.permutations(range(4)))
PERMUTATION_INDEX = {permutation: index for index, permutation in enumerate(PERMUTATIONS)}
SEED = 20260806
EPOCHS = 56
BATCH_SIZE = 96
INFERENCE_BATCH_SIZE = 192
NUM_WORKERS = 2
EXPECTED_TRAIN_USERS = {'user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user16', 'user17', 'user18', 'user19', 'user20', 'user21', 'user22', 'user23', 'user24'}
EXPECTED_TRAIN_UNITS = 3912
EXPECTED_MODEL_TRAIN_UNITS = 3907
EXPECTED_UNSUPERVISED_HAU_UNITS = 5
EXPECTED_TRAIN_ROWS = 4087
EXPECTED_TRAIN_QA_UNITS = 1333
EXPECTED_TEST_UNITS = 208
EXPECTED_HAU_TEST_UNITS = 144
EXPECTED_HARN_TEST_UNITS = 64
EXPECTED_HARN_SINGLE_UNITS = 51
EXPECTED_HARN_OBJECT_ONLY_UNITS = 13
EXPECTED_SENSOR_CACHE_UNITS = 3932
EXPECTED_VISION_CACHE_UNITS = 4120
EXPECTED_TEST_ROWS = 682
EXPECTED_ACTION_CLASSES = 45
EXPECTED_STEPS = 8
EXPECTED_SENSOR_WIDTH = 263
EXPECTED_SENSOR_GLOBAL_WIDTH = 1055
EXPECTED_MOTION_WIDTH = 2107
EXPECTED_VISION_WIDTH = 1152
EXPECTED_SIGLIP_REVISION = 'dd658faac399427308559e2c3ac1e99cbe43845d'
TEST_MIX = {('HAU', 'single'): 144, ('HAU', 'multi'): 144, ('HAU', 'combination'): 139, ('HAU', 'emotion'): 144, ('HAU', 'sequence'): 39, ('HARn', 'single'): 51, ('HARn', 'object_interaction'): 21}
TEST_OPTION_COUNTS = {('HAU', 'single'): 4, ('HAU', 'multi'): 4, ('HAU', 'combination'): 4, ('HAU', 'emotion'): 4, ('HAU', 'sequence'): 4, ('HARn', 'single'): 3, ('HARn', 'object_interaction'): 4}
TARGET_CATEGORIES = ('single', 'multi', 'combination', 'sequence')
MIN_RULE_SUPPORT = 20
MIN_RULE_ACCURACY = 0.995
RAW_ANCHOR_PRIOR = 0.6
PAIR_FACTOR_WEIGHT = 0.35
CATEGORY_PRIOR_WEIGHT = 0.35
LAPLACE_ALPHA = 1.0
ALIASES = {'put on clothes': 'getting dressed', 'putting on clothes': 'getting dressed', 'take off clothes': 'undressing', 'taking off clothes': 'undressing', 'take and use tableware': 'grabbing utensils', 'taking and using tableware': 'grabbing utensils', 'turn pages': 'turning a page', 'turning pages': 'turning a page', 'take body temperature': 'checking body temperature', 'taking body temperature': 'checking body temperature', 'make a phone call': 'calling', 'making a phone call': 'calling', 'use a mobile phone': 'using a phone', 'using a mobile phone': 'using a phone', 'tap the keyboard': 'typing on a keyboard', 'tapping the keyboard': 'typing on a keyboard', 'jog in place': 'running', 'jogging in place': 'running', 'do squats': 'squats', 'doing squats': 'squats', 'do lunges': 'lunges', 'doing lunges': 'lunges', 'do jumping jacks': 'jumping jacks', 'doing jumping jacks': 'jumping jacks', 'do stretching exercises': 'stretching', 'doing stretching exercises': 'stretching', 'listen to music with headphones': 'listening to the music with headphones', 'listening to music with headphones': 'listening to the music with headphones', 'watch tv': 'watching tv', 'play games': 'playing games', 'drink water': 'drinking water', 'drinking': 'drinking water', 'eat food': 'eating', 'peel fruits': 'peeling fruit', 'wipe bowls': 'wiping bowls', 'wipe hands': 'wiping hands', 'wipe windows and tables': 'wiping surface', 'wash face': 'washing face', 'comb hair': 'combing hair', 'brush teeth': 'brushing teeth', 'fold clothes': 'folding clothes', 'mop the floor': 'mopping', 'sweep the floor': 'sweeping', 'pour drinks': 'pouring', 'stir drinks': 'stirring', 'read documents': 'reading', 'write': 'writing', 'stand up': 'standing up', 'sit down': 'sitting down', 'lie down': 'lying down', 'take a selfie': 'taking a selfie', 'take medicine': 'taking medicine', 'massage oneself': 'massaging oneself', 'check the time': 'checking the time', 'walking': 'walk'}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

def norm_text(value: object) -> str:
    text = str(value or '').strip().lower().replace('&', ' and ')
    text = re.sub('[^a-z0-9]+', ' ', text)
    return re.sub('\\s+', ' ', text).strip()
NORMALIZED_ALIASES = {norm_text(key): norm_text(value) for key, value in ALIASES.items()}

def canonical_action(value: object, learned: dict[str, str | tuple[str, ...]] | None=None) -> str:
    normalized = norm_text(value)
    if learned and normalized in learned:
        resolved = learned[normalized]
        if isinstance(resolved, str):
            return resolved
        if len(resolved) == 1:
            return resolved[0]
        return NORMALIZED_ALIASES.get(normalized, normalized)
    return NORMALIZED_ALIASES.get(normalized, normalized)

def split_atoms(value: object) -> list[str]:
    return [part.strip() for part in str(value or '').split(',') if part.strip()]

def find_input_file(name: str, required_fragment: str) -> Path:
    matches = [path for path in INPUT_ROOT.rglob(name) if required_fragment.lower() in str(path).lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one {name!r} containing {required_fragment!r}; found {matches}')
    return matches[0]

def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding='utf-8'))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def training_unit_from_path(value: object) -> str | None:
    path = str(value).replace('\\', '/').strip('/')
    match = re.match('(HAU/user\\d+/[^/]+)', path, re.I)
    if match:
        parts = match.group(1).split('/')
        parts[0] = 'HAU'
        parts[1] = parts[1].lower()
        return '/'.join(parts)
    match = re.match('(HARn/[^/]+/user\\d+/[^/]+)', path, re.I)
    if match:
        parts = match.group(1).split('/')
        parts[0] = 'HARn'
        parts[2] = parts[2].lower()
        return '/'.join(parts)
    return None

def test_unit_from_path(value: object) -> str | None:
    path = str(value).replace('\\', '/')
    match = re.search('(?:^|/)(large_model_track_test/(LM_test_\\d+))(?:/|$)', path, re.I)
    if not match:
        return None
    return f'large_model_track_test/{match.group(2)}'

def user_from_unit(unit: object) -> str | None:
    match = re.search('/(user\\d+)/', str(unit), re.I)
    return match.group(1).lower() if match else None

def harn_folder(unit: object) -> str | None:
    parts = str(unit).split('/')
    return parts[1] if len(parts) >= 4 and parts[0].lower() == 'harn' else None

def available_letters(row: pd.Series) -> str:
    return ''.join((letter for letter in LETTERS if str(row.get(letter, '')).strip()))

def candidate_values(row: pd.Series) -> list[str]:
    letters = available_letters(row)
    category = str(row['category'])
    if category == 'multi':
        return [''.join((letters[index] for index in range(len(letters)) if mask & 1 << index)) for mask in range(1, 1 << len(letters))]
    if category == 'sequence':
        return [''.join(permutation) for permutation in itertools.permutations(letters)]
    return list(letters)

def clean_prediction(value: object, row: pd.Series) -> str | None:
    letters = available_letters(row)
    selected = ''.join((character for character in str(value or '').upper() if character in letters))
    selected = ''.join(dict.fromkeys(selected))
    category = str(row['category'])
    if category == 'multi':
        selected = ''.join((letter for letter in letters if letter in selected))
    elif category != 'sequence':
        selected = selected[:1]
    return selected if selected in set(candidate_values(row)) else None

def correct_letter(row: pd.Series) -> str | None:
    answer = clean_prediction(row['answer'], row)
    return answer if answer and len(answer) == 1 else None

def correct_option(row: pd.Series) -> str | None:
    letter = correct_letter(row)
    return str(row[letter]) if letter else None

def validate_input_tables(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_columns = ['qa_id', 'source', 'path', 'category', 'question', *LETTERS, 'answer']
    test_columns = ['qa_id', 'source', 'path', 'category', 'question', *LETTERS]
    missing_train = sorted(set(train_columns) - set(train_raw.columns))
    missing_test = sorted(set(test_columns) - set(test_raw.columns))
    if missing_train or missing_test:
        raise AssertionError(f'Missing competition columns: train={missing_train}, test={missing_test}')
    for forbidden in ('answer', 'prediction'):
        if forbidden in test_raw.columns and test_raw[forbidden].fillna('').astype(str).str.strip().ne('').any():
            raise AssertionError(f'Official test column {forbidden!r} unexpectedly contains values')
    train = train_raw[train_columns].fillna('').astype(str).copy()
    test = test_raw[test_columns].fillna('').astype(str).copy()
    if not train['qa_id'].is_unique or not test['qa_id'].is_unique:
        raise AssertionError('Competition qa_id values must be unique within each split')
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise AssertionError(f'Expected {EXPECTED_TRAIN_ROWS} training QA rows, found {len(train)}')
    if set(train['qa_id']) & set(test['qa_id']):
        raise AssertionError('Training and test qa_id values overlap')
    if len(test) != EXPECTED_TEST_ROWS:
        raise AssertionError(f'Expected {EXPECTED_TEST_ROWS} test rows, found {len(test)}')
    if not test['qa_id'].str.fullmatch('test_\\d+').all():
        raise AssertionError('Anonymous test IDs do not match the official test_#### grammar')
    train['unit'] = train['path'].map(training_unit_from_path)
    train['user'] = train['unit'].map(user_from_unit)
    test['unit'] = test['path'].map(test_unit_from_path)
    if train['unit'].isna().any() or train['user'].isna().any():
        raise AssertionError('Could not parse one or more canonical training units/users')
    path_unit_pairs = train[['path', 'unit']].drop_duplicates()
    if path_unit_pairs.groupby('unit')['path'].nunique().max() != 1 or path_unit_pairs.groupby('path')['unit'].nunique().max() != 1 or path_unit_pairs['path'].nunique() != path_unit_pairs['unit'].nunique():
        raise AssertionError('Canonical training units are not one-to-one with the exact path groups used by the promoted graph-v2 parent')
    if train['unit'].nunique() != EXPECTED_TRAIN_QA_UNITS:
        raise AssertionError(f"Expected {EXPECTED_TRAIN_QA_UNITS} QA-bearing training units, found {train['unit'].nunique()}")
    expected_train_source = train['unit'].map(lambda unit: 'HAU' if str(unit).startswith('HAU/') else 'HARn' if str(unit).startswith('HARn/') else None)
    if expected_train_source.isna().any() or not expected_train_source.eq(train['source']).all():
        raise AssertionError('Official training source values disagree with canonical unit prefixes')
    if test['unit'].isna().any():
        raise AssertionError('Could not parse one or more anonymous official test units')
    test_path_unit_pairs = test[['path', 'unit']].drop_duplicates()
    if test_path_unit_pairs.groupby('unit')['path'].nunique().max() != 1 or test_path_unit_pairs.groupby('path')['unit'].nunique().max() != 1 or test_path_unit_pairs['path'].nunique() != test_path_unit_pairs['unit'].nunique():
        raise AssertionError('Anonymous canonical units are not one-to-one with exact test path groups')
    if test.groupby(['unit', 'category']).size().max() != 1:
        raise AssertionError('An anonymous unit has duplicate rows for one category')
    observed_users = set(train['user'])
    if observed_users != EXPECTED_TRAIN_USERS:
        raise AssertionError(f'Canonical training users changed: expected={sorted(EXPECTED_TRAIN_USERS)}, observed={sorted(observed_users)}')
    if any((user_from_unit(unit) for unit in test['unit'])):
        raise AssertionError('A test unit unexpectedly contains a training user identifier')
    test_mix = test.groupby(['source', 'category']).size().to_dict()
    if test_mix != TEST_MIX:
        raise AssertionError(f'Official test composition changed: {test_mix}')
    for _, row in test.iterrows():
        key = (str(row['source']), str(row['category']))
        expected_options = TEST_OPTION_COUNTS[key]
        observed_options = len(available_letters(row))
        if observed_options != expected_options:
            raise AssertionError(f"Option-count contract changed for {row['qa_id']}: expected={expected_options}, observed={observed_options}")
    if test['unit'].nunique() != EXPECTED_TEST_UNITS:
        raise AssertionError(f"Expected {EXPECTED_TEST_UNITS} anonymous test units, found {test['unit'].nunique()}")
    if test.groupby('unit')['source'].nunique().max() != 1:
        raise AssertionError('An anonymous test unit occurs under multiple sources')
    source_counts = test[['unit', 'source']].drop_duplicates().groupby('source').size().to_dict()
    expected_source_counts = {'HAU': EXPECTED_HAU_TEST_UNITS, 'HARn': EXPECTED_HARN_TEST_UNITS}
    if source_counts != expected_source_counts:
        raise AssertionError(f'Official test unit/source composition changed: {source_counts}')
    harn = test[test['source'].eq('HARn')]
    harn_single_units = set(harn.loc[harn['category'].eq('single'), 'unit'])
    harn_object_units = set(harn.loc[harn['category'].eq('object_interaction'), 'unit'])
    harn_object_only_units = harn_object_units - harn_single_units
    if len(harn_single_units) != EXPECTED_HARN_SINGLE_UNITS:
        raise AssertionError(f'Expected {EXPECTED_HARN_SINGLE_UNITS} HARn single units, found {len(harn_single_units)}')
    if len(harn_object_only_units) != EXPECTED_HARN_OBJECT_ONLY_UNITS:
        raise AssertionError(f'Expected {EXPECTED_HARN_OBJECT_ONLY_UNITS} object-only HARn units, found {len(harn_object_only_units)}')
    return (train, test)

def load_and_align_features(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    sensor_npz_path = find_input_file('sensor_features.npz', 'cuhk-x-lmt-sensor-feature-cache')
    sensor_units_path = find_input_file('sensor_units.json', 'cuhk-x-lmt-sensor-feature-cache')
    sensor_manifest_path = find_input_file('sensor_feature_manifest.json', 'cuhk-x-lmt-sensor-feature-cache')
    vision_npz_path = find_input_file('siglip_video_features.npz', 'cuhk-x-siglip2-so400m-video-feature-cache')
    vision_units_path = find_input_file('siglip_video_units.json', 'cuhk-x-siglip2-so400m-video-feature-cache')
    vision_manifest_path = find_input_file('siglip_feature_manifest.json', 'cuhk-x-siglip2-so400m-video-feature-cache')
    sensor_manifest = read_json(sensor_manifest_path)
    vision_manifest = read_json(vision_manifest_path)
    if sensor_manifest.get('errors'):
        raise AssertionError(f"Sensor cache reported errors: {sensor_manifest['errors'][:5]}")
    if vision_manifest.get('errors'):
        raise AssertionError(f"SigLIP cache reported errors: {vision_manifest['errors'][:5]}")
    if int(sensor_manifest.get('bins', -1)) != EXPECTED_STEPS:
        raise AssertionError('Sensor cache temporal-bin contract changed')
    if int(sensor_manifest.get('segment_width', -1)) != EXPECTED_SENSOR_WIDTH:
        raise AssertionError('Sensor cache segment-width contract changed')
    if int(vision_manifest.get('frames_per_unit', -1)) != EXPECTED_STEPS:
        raise AssertionError('SigLIP cache frame-count contract changed')
    if vision_manifest.get('resolved_revision') != EXPECTED_SIGLIP_REVISION:
        raise AssertionError('SigLIP cache model revision changed')
    if int(vision_manifest.get('valid_unit_count', -1)) != int(vision_manifest.get('unit_count', -2)):
        raise AssertionError('SigLIP cache contains invalid units')
    with np.load(sensor_npz_path) as sensor_bundle:
        if 'segments' not in sensor_bundle.files or 'global_features' not in sensor_bundle.files:
            raise AssertionError('Sensor cache is missing the masked-MIL-v2 segments/global_features tensors')
        sensor_tokens = sensor_bundle['segments'].astype(np.float32)
        sensor_global = sensor_bundle['global_features'].astype(np.float32)
    with np.load(vision_npz_path) as vision_bundle:
        if 'frame_embeddings' not in vision_bundle.files:
            raise AssertionError('SigLIP cache is missing frame_embeddings')
        vision_tokens = vision_bundle['frame_embeddings'].astype(np.float32)
    sensor_units = [str(unit) for unit in read_json(sensor_units_path)]
    vision_units = [str(unit) for unit in read_json(vision_units_path)]
    if len(sensor_units) != EXPECTED_SENSOR_CACHE_UNITS:
        raise AssertionError(f'Expected {EXPECTED_SENSOR_CACHE_UNITS} sensor units, found {len(sensor_units)}')
    if len(vision_units) != EXPECTED_VISION_CACHE_UNITS:
        raise AssertionError(f'Expected {EXPECTED_VISION_CACHE_UNITS} SigLIP units, found {len(vision_units)}')
    if len(sensor_units) != len(set(sensor_units)) or len(vision_units) != len(set(vision_units)):
        raise AssertionError('Feature-cache unit identifiers are not unique')
    if sensor_tokens.ndim != 3 or vision_tokens.ndim != 3:
        raise AssertionError('Feature tensors must be [unit, step, feature]')
    if sensor_global.ndim != 2:
        raise AssertionError('Sensor global_features must be [unit, feature]')
    if sensor_tokens.shape != (len(sensor_units), EXPECTED_STEPS, EXPECTED_SENSOR_WIDTH):
        raise AssertionError(f'Unexpected sensor tensor shape {sensor_tokens.shape} for {len(sensor_units)} units')
    if vision_tokens.shape != (len(vision_units), EXPECTED_STEPS, EXPECTED_VISION_WIDTH):
        raise AssertionError(f'Unexpected vision tensor shape {vision_tokens.shape} for {len(vision_units)} units')
    if sensor_global.shape != (len(sensor_units), EXPECTED_SENSOR_GLOBAL_WIDTH):
        raise AssertionError(f'Unexpected sensor global tensor shape {sensor_global.shape}')
    if not np.isfinite(sensor_tokens).all() or not np.isfinite(sensor_global).all() or (not np.isfinite(vision_tokens).all()):
        raise AssertionError('Feature caches contain NaN or infinite values')
    sensor_index = {unit: index for index, unit in enumerate(sensor_units)}
    vision_index = {unit: index for index, unit in enumerate(vision_units)}
    all_units = sorted(set(sensor_units) | set(vision_units))
    unit_index = {unit: index for index, unit in enumerate(all_units)}
    feature_train_units = sorted((unit for unit in all_units if user_from_unit(unit) in EXPECTED_TRAIN_USERS))
    test_units = sorted(set(test['unit']))
    if len(feature_train_units) != EXPECTED_TRAIN_UNITS:
        raise AssertionError(f'Expected {EXPECTED_TRAIN_UNITS} canonical training feature units, found {len(feature_train_units)}')
    if len(test_units) != EXPECTED_TEST_UNITS:
        raise AssertionError(f'Expected {EXPECTED_TEST_UNITS} anonymous test units, found {len(test_units)}')
    training_qa_units = set(train['unit'])
    if not training_qa_units <= set(feature_train_units):
        missing = sorted(training_qa_units - set(feature_train_units))
        raise AssertionError(f'Training QA units are absent from feature caches: {missing[:10]}')
    unsupervised_non_harn = sorted((unit for unit in feature_train_units if harn_folder(unit) is None and unit not in training_qa_units))
    if len(unsupervised_non_harn) != EXPECTED_UNSUPERVISED_HAU_UNITS:
        raise AssertionError(f'Expected {EXPECTED_UNSUPERVISED_HAU_UNITS} label-free HAU feature units, found {len(unsupervised_non_harn)}: {unsupervised_non_harn[:10]}')
    train_units = sorted(set(feature_train_units) - set(unsupervised_non_harn))
    if len(train_units) != EXPECTED_MODEL_TRAIN_UNITS:
        raise AssertionError(f'Expected {EXPECTED_MODEL_TRAIN_UNITS} supervised model-training units, found {len(train_units)}')
    if not set(test_units) <= set(all_units):
        raise AssertionError('One or more official test units are absent from both feature caches')
    if set(feature_train_units) & set(test_units):
        raise AssertionError('Canonical training and anonymous test feature units overlap')
    unclassified = sorted(set(all_units) - set(feature_train_units) - set(test_units))
    if unclassified:
        raise AssertionError(f'Feature caches contain unexpected units: {unclassified[:10]}')
    source_ids = np.full(len(all_units), -1, dtype=np.int64)
    for unit in feature_train_units:
        if unit.startswith('HAU/'):
            source_ids[unit_index[unit]] = 0
        elif unit.startswith('HARn/'):
            source_ids[unit_index[unit]] = 1
        else:
            raise AssertionError(f'Canonical training feature unit has unknown source prefix: {unit}')
    test_unit_sources = test[['unit', 'source']].drop_duplicates()
    if len(test_unit_sources) != EXPECTED_TEST_UNITS:
        raise AssertionError('Anonymous unit/source mapping is not one row per unit')
    for record in test_unit_sources.itertuples(index=False):
        if record.source == 'HAU':
            source_id = 0
        elif record.source == 'HARn':
            source_id = 1
        else:
            raise AssertionError(f'Anonymous unit {record.unit} has unknown source {record.source!r}')
        source_ids[unit_index[str(record.unit)]] = source_id
    if not np.isin(source_ids, [0, 1]).all():
        missing = [unit for unit, index in unit_index.items() if source_ids[index] not in (0, 1)]
        raise AssertionError(f'Feature units lack validated source routing: {missing[:10]}')
    test_source_counts = Counter((int(source_ids[unit_index[unit]]) for unit in test_units))
    if test_source_counts != Counter({0: EXPECTED_HAU_TEST_UNITS, 1: EXPECTED_HARN_TEST_UNITS}):
        raise AssertionError(f'Anonymous source-embedding routing changed: {dict(test_source_counts)}')
    sensor_aligned = np.zeros((len(all_units), EXPECTED_STEPS, EXPECTED_SENSOR_WIDTH), dtype=np.float32)
    sensor_global_aligned = np.zeros((len(all_units), EXPECTED_SENSOR_GLOBAL_WIDTH), dtype=np.float32)
    vision_aligned = np.zeros((len(all_units), EXPECTED_STEPS, EXPECTED_VISION_WIDTH), dtype=np.float32)
    modality_mask = np.zeros((len(all_units), 2), dtype=np.float32)
    for index, unit in enumerate(all_units):
        sensor_position = sensor_index.get(unit)
        vision_position = vision_index.get(unit)
        if sensor_position is not None:
            sensor_aligned[index] = sensor_tokens[sensor_position]
            sensor_global_aligned[index] = sensor_global[sensor_position]
            modality_mask[index, 0] = 1.0
        if vision_position is not None:
            vision_aligned[index] = vision_tokens[vision_position]
            modality_mask[index, 1] = 1.0
    if not np.all(modality_mask[:, 1] == 1.0):
        missing_vision = [all_units[index] for index in np.where(modality_mask[:, 1] == 0.0)[0]]
        raise AssertionError(f'Depth_Color SigLIP coverage is incomplete: {missing_vision[:10]}')
    if not np.all(modality_mask.sum(axis=1) >= 1.0):
        raise AssertionError('A feature unit has no usable modality')
    sensor_delta = np.diff(sensor_aligned, axis=1)
    motion_raw = np.concatenate([sensor_global_aligned, sensor_aligned.std(axis=1), np.abs(sensor_delta).mean(axis=1), np.abs(sensor_delta).std(axis=1), np.abs(sensor_delta).max(axis=1)], axis=1).astype(np.float32)
    if motion_raw.shape != (len(all_units), EXPECTED_MOTION_WIDTH):
        raise AssertionError(f'Masked-MIL-v2 motion width changed: {motion_raw.shape}')
    train_positions = np.asarray([unit_index[unit] for unit in train_units], dtype=np.int64)
    sensor_valid = modality_mask[train_positions, 0] > 0.5
    fit_motion = motion_raw[train_positions[sensor_valid]]
    if len(fit_motion) == 0:
        raise AssertionError('No supervised training units have sensor features')
    motion_mean = fit_motion.mean(axis=0, dtype=np.float64).astype(np.float32)
    motion_std = fit_motion.std(axis=0, dtype=np.float64).astype(np.float32)
    motion_std = np.where(motion_std > 1e-05, motion_std, 1.0).astype(np.float32)
    motion_scaled = np.clip((motion_raw - motion_mean) / motion_std, -8.0, 8.0).astype(np.float32)
    motion_scaled[modality_mask[:, 0] < 0.5] = 0.0
    if not np.isfinite(motion_scaled).all():
        raise AssertionError('Scaled masked-MIL-v2 motion features are non-finite')
    return {'all_units': all_units, 'feature_train_units': feature_train_units, 'train_units': train_units, 'test_units': test_units, 'unit_index': unit_index, 'sensor': sensor_aligned, 'vision': vision_aligned, 'motion': motion_scaled, 'modality': modality_mask, 'source_ids': source_ids, 'sensor_manifest': sensor_manifest, 'vision_manifest': vision_manifest, 'cache_audit': {'sensor_cache_units': len(sensor_units), 'vision_cache_units': len(vision_units), 'cache_union_units': len(all_units), 'canonical_train_feature_units': len(feature_train_units), 'supervised_model_train_units': len(train_units), 'excluded_label_free_hau_units': unsupervised_non_harn, 'training_qa_rows': len(train), 'training_qa_units': len(training_qa_units), 'folder_supervised_harn_units': sum((harn_folder(unit) is not None for unit in feature_train_units)), 'anonymous_inference_units': len(test_units), 'hau_inference_units': int(test.loc[test['source'].eq('HAU'), 'unit'].nunique()), 'harn_inference_units': int(test.loc[test['source'].eq('HARn'), 'unit'].nunique()), 'harn_single_units': int(test.loc[test['source'].eq('HARn') & test['category'].eq('single'), 'unit'].nunique()), 'harn_object_only_units': len(set(test.loc[test['source'].eq('HARn') & test['category'].eq('object_interaction'), 'unit']) - set(test.loc[test['source'].eq('HARn') & test['category'].eq('single'), 'unit'])), 'source_embedding_training_route': 'canonical HAU/HARn unit prefix', 'source_embedding_test_route': 'official safe source column', 'source_embedding_test_hau_units': int(test_source_counts[0]), 'source_embedding_test_harn_units': int(test_source_counts[1]), 'motion_width': EXPECTED_MOTION_WIDTH, 'motion_fit_sensor_units': int(sensor_valid.sum()), 'motion_constant_channels': int(np.sum(motion_std == 1.0)), 'motion_scaler_fit_on_supervised_training_only': True}, 'resource_paths': {'sensor_npz': str(sensor_npz_path), 'sensor_units': str(sensor_units_path), 'sensor_manifest': str(sensor_manifest_path), 'vision_npz': str(vision_npz_path), 'vision_units': str(vision_units_path), 'vision_manifest': str(vision_manifest_path)}, 'resource_hashes': {'sensor_npz_sha256': sha256_file(sensor_npz_path), 'sensor_units_sha256': sha256_file(sensor_units_path), 'sensor_manifest_sha256': sha256_file(sensor_manifest_path), 'vision_npz_sha256': sha256_file(vision_npz_path), 'vision_units_sha256': sha256_file(vision_units_path), 'vision_manifest_sha256': sha256_file(vision_manifest_path)}}

def build_ontology(train: pd.DataFrame, train_units: list[str]) -> dict:
    class_names = set()
    for unit in train_units:
        folder = harn_folder(unit)
        if folder:
            class_names.add(canonical_action(re.sub('^\\d+_', '', folder).replace('_', ' ')))
    action_rows = train[train['source'].eq('HAU') & train['category'].isin(['single', 'multi', 'combination', 'sequence'])]
    for _, row in action_rows.iterrows():
        for letter in available_letters(row):
            values = split_atoms(row[letter]) if row['category'] == 'combination' else [row[letter]]
            for value in values:
                if norm_text(value):
                    class_names.add(canonical_action(value))
    class_names = sorted(class_names)
    if len(class_names) != EXPECTED_ACTION_CLASSES:
        raise AssertionError(f'Expected {EXPECTED_ACTION_CLASSES} action classes, found {len(class_names)}')
    class_index = {name: index for index, name in enumerate(class_names)}
    emotion_values = sorted({norm_text(row[letter]) for _, row in train[train['category'].eq('emotion')].iterrows() for letter in available_letters(row) if norm_text(row[letter])})
    object_values = sorted({norm_text(row[letter]) for _, row in train[train['category'].eq('object_interaction')].iterrows() for letter in available_letters(row) if norm_text(row[letter])})
    emotion_vocab = {'<unk>': 0, **{value: index + 1 for index, value in enumerate(emotion_values)}}
    object_vocab = {'<unk>': 0, **{value: index + 1 for index, value in enumerate(object_values)}}
    return {'class_names': class_names, 'class_index': class_index, 'emotion_vocab': emotion_vocab, 'object_vocab': object_vocab}

def option_ids(row: pd.Series, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(norm_text(row.get(letter, '')), 0) for letter in LETTERS]

def build_fullfit_aliases(train: pd.DataFrame) -> dict[str, str | tuple[str, ...]]:
    candidates = defaultdict(Counter)
    subset = train[train['source'].eq('HARn') & train['category'].eq('single')]
    for _, row in subset.iterrows():
        value = correct_option(row)
        folder = harn_folder(row['unit'])
        if value and folder:
            semantic = canonical_action(re.sub('^\\d+_', '', folder).replace('_', ' '))
            normalized = norm_text(value)
            candidates[normalized][semantic] += 1
    return {normalized: next(iter(counts)) if len(counts) == 1 else tuple(sorted(counts)) for normalized, counts in candidates.items()}

def hashed_character_features(values: list[str], width: int=48) -> np.ndarray:
    features = np.zeros((len(values), width), dtype=np.float32)
    for row_index, value in enumerate(values):
        text = f'^{norm_text(value)}$'
        for ngram_size in (2, 3, 4, 5):
            for start in range(max(0, len(text) - ngram_size + 1)):
                token = text[start:start + ngram_size].encode('utf-8')
                digest = hashlib.blake2b(token, digest_size=8).digest()
                number = int.from_bytes(digest, 'little')
                bucket = number % width
                sign = 1.0 if number >> 8 & 1 else -1.0
                features[row_index, bucket] += sign / math.sqrt(ngram_size)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-08)

def build_emotion_semantics(fit_rows: pd.DataFrame, emotion_vocab: dict[str, int]) -> tuple[np.ndarray, dict]:
    """Exact v2 offline text/topology initialization fitted on training only."""
    values_by_id = [value for value, _ in sorted(emotion_vocab.items(), key=lambda item: item[1])]
    vocab_size = len(values_by_id)
    character = hashed_character_features(values_by_id)
    character[0] = 0.0
    cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    wins = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    losses = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    occurrence = np.zeros(vocab_size, dtype=np.float32)
    correct_count = np.zeros(vocab_size, dtype=np.float32)
    position_sum = np.zeros(vocab_size, dtype=np.float32)
    emotion_rows = fit_rows[fit_rows['category'].eq('emotion')]
    for _, row in emotion_rows.iterrows():
        ids = option_ids(row, emotion_vocab)
        letter = correct_letter(row)
        if not letter:
            continue
        correct_id = ids[LETTERS.index(letter)]
        for position, option_id in enumerate(ids):
            occurrence[option_id] += 1.0
            position_sum[option_id] += position / 3.0
            for other_id in ids:
                if option_id != other_id:
                    cooccurrence[option_id, other_id] += 1.0
        correct_count[correct_id] += 1.0
        for option_id in ids:
            if option_id != correct_id:
                wins[correct_id, option_id] += 1.0
                losses[option_id, correct_id] += 1.0
    graph_profile = np.concatenate([np.log1p(cooccurrence), np.log1p(wins), np.log1p(losses)], axis=1)
    graph_profile[0] = 0.0
    graph_rank = min(16, max(1, min(graph_profile.shape) - 1))
    try:
        u, singular, _ = np.linalg.svd(graph_profile, full_matrices=False)
        graph = (u[:, :graph_rank] * singular[:graph_rank]).astype(np.float32)
    except np.linalg.LinAlgError:
        graph = graph_profile[:, :graph_rank].astype(np.float32)
    graph /= np.maximum(np.linalg.norm(graph, axis=1, keepdims=True), 1e-08)
    maximum_occurrence = max(float(occurrence.max()), 1.0)
    statistics = np.stack([np.log1p(occurrence) / math.log1p(maximum_occurrence), correct_count / np.maximum(occurrence, 1.0), position_sum / np.maximum(occurrence, 1.0), (occurrence > 0).astype(np.float32)], axis=1)
    semantics = np.concatenate([character, graph, statistics], axis=1).astype(np.float32)
    semantics[0] = 0.0
    semantics /= np.maximum(np.linalg.norm(semantics, axis=1, keepdims=True), 1e-08)
    return (semantics, {'method': 'offline character hashing plus full-train option/win/loss graph SVD', 'fit_emotion_rows': int(len(emotion_rows)), 'vocabulary_size': int(vocab_size), 'embedding_width': int(semantics.shape[1]), 'graph_rank': int(graph_rank), 'unseen_fit_options': [values_by_id[index] for index in range(1, vocab_size) if occurrence[index] == 0]})

def empty_target(n_actions: int) -> dict:
    return {'action': np.zeros(n_actions, dtype=np.float32), 'known': np.zeros(n_actions, dtype=np.float32), 'order': np.zeros((n_actions, n_actions), dtype=np.float32), 'harn': -1, 'multi_options': np.full(4, -1, dtype=np.int64), 'multi_count': -1, 'sequence_options': np.full(4, -1, dtype=np.int64), 'sequence_target': -1, 'emotion_options': np.zeros(4, dtype=np.int64), 'emotion_target': -1, 'object_options': np.zeros(4, dtype=np.int64), 'object_target': -1}

def set_action_label(record: dict, action_index: int, value: float, unit: str, class_name: str, conflicts: list[dict]) -> None:
    if record['known'][action_index] and record['action'][action_index] != value:
        conflicts.append({'unit': unit, 'class': class_name, 'old': float(record['action'][action_index]), 'new': float(value)})
    record['action'][action_index] = value
    record['known'][action_index] = 1.0

def build_targets(train: pd.DataFrame, test: pd.DataFrame, all_units: list[str], train_units: list[str], ontology: dict, aliases: dict[str, str]) -> tuple[dict[str, dict], dict]:
    class_names = ontology['class_names']
    class_index = ontology['class_index']
    emotion_vocab = ontology['emotion_vocab']
    object_vocab = ontology['object_vocab']
    targets = {unit: empty_target(len(class_names)) for unit in all_units}
    conflicts = []
    for unit in train_units:
        folder = harn_folder(unit)
        if folder:
            semantic = canonical_action(re.sub('^\\d+_', '', folder).replace('_', ' '))
            if semantic not in class_index:
                raise AssertionError(f'HARn folder action {semantic!r} is outside the ontology')
            targets[unit]['harn'] = class_index[semantic]
    for _, row in train.iterrows():
        unit = row['unit']
        record = targets[unit]
        category = str(row['category'])
        answer = clean_prediction(row['answer'], row)
        if answer is None:
            raise AssertionError(f"Invalid training answer grammar for {row['qa_id']}")
        if row['source'] == 'HAU' and category in {'single', 'multi'}:
            selected = set(answer)
            option_action_ids = []
            for letter in available_letters(row):
                semantic = canonical_action(row[letter], aliases)
                action_index = class_index.get(semantic, -1)
                option_action_ids.append(action_index)
                if action_index >= 0:
                    set_action_label(record, action_index, float(letter in selected), unit, semantic, conflicts)
            if category == 'multi' and len(option_action_ids) == 4 and all((action_index >= 0 for action_index in option_action_ids)):
                record['multi_options'] = np.asarray(option_action_ids, dtype=np.int64)
                record['multi_count'] = len(selected)
        elif row['source'] == 'HAU' and category == 'combination':
            for atom in split_atoms(row[answer]):
                semantic = canonical_action(atom, aliases)
                if semantic in class_index:
                    action_index = class_index[semantic]
                    set_action_label(record, action_index, 1.0, unit, semantic, conflicts)
        elif row['source'] == 'HAU' and category == 'sequence':
            option_action_ids = [class_index.get(canonical_action(row[letter], aliases), -1) for letter in available_letters(row)]
            if len(option_action_ids) == 4 and all((action_index >= 0 for action_index in option_action_ids)):
                record['sequence_options'] = np.asarray(option_action_ids, dtype=np.int64)
                permutation = tuple((LETTERS.index(letter) for letter in answer))
                if permutation in PERMUTATION_INDEX:
                    record['sequence_target'] = PERMUTATION_INDEX[permutation]
            ordered = []
            for letter in answer:
                semantic = canonical_action(row[letter], aliases)
                if semantic in class_index:
                    action_index = class_index[semantic]
                    ordered.append(action_index)
                    set_action_label(record, action_index, 1.0, unit, semantic, conflicts)
            for left_position, left_action in enumerate(ordered):
                for right_action in ordered[left_position + 1:]:
                    record['order'][left_action, right_action] = 1.0
        elif category == 'emotion':
            record['emotion_options'] = np.asarray(option_ids(row, emotion_vocab), dtype=np.int64)
            record['emotion_target'] = LETTERS.index(answer)
        elif category == 'object_interaction':
            record['object_options'] = np.asarray(option_ids(row, object_vocab), dtype=np.int64)
            record['object_target'] = LETTERS.index(answer)
    for _, row in test.iterrows():
        record = targets[row['unit']]
        category = str(row['category'])
        if category == 'multi':
            option_action_ids = np.asarray([class_index.get(canonical_action(row[letter], aliases), -1) for letter in available_letters(row)], dtype=np.int64)
            if len(option_action_ids) != 4:
                raise AssertionError(f"Test multi option width changed for {row['qa_id']}")
            record['multi_options'] = option_action_ids
        elif category == 'sequence':
            option_action_ids = np.asarray([class_index.get(canonical_action(row[letter], aliases), -1) for letter in available_letters(row)], dtype=np.int64)
            if len(option_action_ids) != 4:
                raise AssertionError(f"Test sequence option width changed for {row['qa_id']}")
            record['sequence_options'] = option_action_ids
        elif category == 'emotion':
            options = np.asarray(option_ids(row, emotion_vocab), dtype=np.int64)
            if record['emotion_options'].any() and (not np.array_equal(record['emotion_options'], options)):
                raise AssertionError(f"Conflicting test emotion options for {row['unit']}")
            record['emotion_options'] = options
        elif category == 'object_interaction':
            options = np.asarray(option_ids(row, object_vocab), dtype=np.int64)
            if record['object_options'].any() and (not np.array_equal(record['object_options'], options)):
                raise AssertionError(f"Conflicting test object options for {row['unit']}")
            record['object_options'] = options
    if conflicts:
        raise AssertionError(f'Conflicting full-training HAU labels: {conflicts[:10]}')
    unsupervised_units = [unit for unit in train_units if targets[unit]['known'].sum() == 0 and targets[unit]['harn'] < 0 and (targets[unit]['emotion_target'] < 0) and (targets[unit]['object_target'] < 0)]
    if unsupervised_units:
        raise AssertionError(f'Canonical training units lack supervision: {unsupervised_units[:10]}')
    hau_records = [targets[unit] for unit in train_units if unit.startswith('HAU/')]
    target_audit = {'classes': class_names, 'class_count': len(class_names), 'train_units': len(train_units), 'supervised_train_units': len(train_units) - len(unsupervised_units), 'hau_units': len(hau_records), 'hau_known_labels': int(sum((record['known'].sum() for record in hau_records))), 'hau_positive_labels': int(sum(((record['action'] * record['known']).sum() for record in hau_records))), 'hau_negative_labels': int(sum((((1.0 - record['action']) * record['known']).sum() for record in hau_records))), 'sequence_pairs': int(sum((record['order'].sum() for record in hau_records))), 'multi_count_targets': int(sum((record['multi_count'] >= 1 for record in hau_records))), 'sequence_listwise_targets': int(sum((record['sequence_target'] >= 0 for record in hau_records))), 'label_conflicts': 0, 'test_targets_created': False}
    return (targets, target_audit)

class UnitDataset(Dataset):

    def __init__(self, units: list[str], targets: dict[str, dict], unit_index: dict[str, int], source_ids: np.ndarray, sensor: np.ndarray, vision: np.ndarray, motion: np.ndarray, modality: np.ndarray) -> None:
        self.units = list(units)
        self.targets = targets
        self.unit_index = unit_index
        self.source_ids = source_ids
        self.sensor = sensor
        self.vision = vision
        self.motion = motion
        self.modality = modality

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, position: int) -> dict:
        unit = self.units[position]
        index = self.unit_index[unit]
        record = self.targets[unit]
        return {'unit': unit, 'sensor': torch.from_numpy(self.sensor[index]), 'vision': torch.from_numpy(self.vision[index]), 'motion': torch.from_numpy(self.motion[index]), 'modality': torch.from_numpy(self.modality[index]), 'source_id': torch.tensor(int(self.source_ids[index]), dtype=torch.long), 'action': torch.from_numpy(record['action']), 'known': torch.from_numpy(record['known']), 'order': torch.from_numpy(record['order']), 'harn': torch.tensor(record['harn'], dtype=torch.long), 'multi_options': torch.from_numpy(record['multi_options']), 'multi_count': torch.tensor(record['multi_count'], dtype=torch.long), 'sequence_options': torch.from_numpy(record['sequence_options']), 'sequence_target': torch.tensor(record['sequence_target'], dtype=torch.long), 'emotion_options': torch.from_numpy(record['emotion_options']), 'emotion_target': torch.tensor(record['emotion_target'], dtype=torch.long), 'object_options': torch.from_numpy(record['object_options']), 'object_target': torch.tensor(record['object_target'], dtype=torch.long)}

class MaskedMILV2(nn.Module):

    def __init__(self, sensor_width: int, vision_width: int, n_steps: int, n_actions: int, motion_width: int, emotion_semantics: np.ndarray, emotion_vocab_size: int, object_vocab_size: int) -> None:
        super().__init__()
        width = 192
        pooled_width = width * 2
        option_width = 96
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.sensor_norm = nn.LayerNorm(sensor_width)
        self.vision_norm = nn.LayerNorm(vision_width)
        self.sensor_proj = nn.Linear(sensor_width, width)
        self.vision_proj = nn.Linear(vision_width, width)
        self.gate = nn.Sequential(nn.Linear(width * 2 + 2, width), nn.GELU(), nn.Linear(width, 1))
        self.position = nn.Parameter(torch.randn(1, n_steps, width) * 0.02)
        self.source_embedding = nn.Embedding(2, width)
        temporal_layer = nn.TransformerEncoderLayer(d_model=width, nhead=6, dim_feedforward=width * 3, dropout=0.18, activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.action_head = nn.Linear(width, n_actions)
        self.action_option_embedding = nn.Embedding(n_actions, 32)
        self.motion_encoder = nn.Sequential(nn.Linear(motion_width, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.2), nn.Linear(384, 192), nn.LayerNorm(192), nn.GELU())
        self.cardinality_clip = nn.Linear(pooled_width, 64)
        cardinality_width = 4 + 4 + 8 + 32 + 64 + 2
        self.cardinality_head = nn.Sequential(nn.Linear(cardinality_width, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 4))
        pair_width = 6 + 32 * 2
        self.order_pair_head = nn.Sequential(nn.Linear(pair_width, 96), nn.GELU(), nn.Dropout(0.15), nn.Linear(96, 1))
        semantic_tensor = torch.from_numpy(emotion_semantics.astype(np.float32))
        self.emotion_semantic = nn.Embedding.from_pretrained(semantic_tensor, freeze=True, padding_idx=0)
        self.emotion_semantic_projection = nn.Linear(int(emotion_semantics.shape[1]), option_width, bias=False)
        self.emotion_residual = nn.Embedding(emotion_vocab_size, 32, padding_idx=0)
        self.emotion_residual_projection = nn.Linear(32, option_width, bias=False)
        manner_width = 192 + pooled_width + n_actions + 2
        self.emotion_clip = nn.Sequential(nn.Linear(manner_width, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, option_width))
        self.emotion_score = nn.Sequential(nn.Linear(option_width * 4, 160), nn.GELU(), nn.Dropout(0.2), nn.Linear(160, 1))
        self.object_embedding = nn.Embedding(object_vocab_size, option_width)
        self.object_clip = nn.Linear(pooled_width, option_width)
        self.object_score = nn.Sequential(nn.Linear(option_width * 3, 96), nn.GELU(), nn.Dropout(0.15), nn.Linear(96, 1))

    @staticmethod
    def object_option_scores(clip, options, embedding, clip_projection, scorer):
        option = embedding(options)
        projected = clip_projection(clip).unsqueeze(1).expand_as(option)
        return scorer(torch.cat([projected, option, projected * option], dim=-1)).squeeze(-1)

    def emotion_option_scores(self, manner, options):
        semantic = self.emotion_semantic_projection(self.emotion_semantic(options))
        residual = self.emotion_residual_projection(self.emotion_residual(options))
        option = semantic + residual
        projected = self.emotion_clip(manner).unsqueeze(1).expand_as(option)
        features = torch.cat([projected, option, projected * option, torch.abs(projected - option)], dim=-1)
        return self.emotion_score(features).squeeze(-1)

    def cardinality_scores(self, clip_logits, pooled, modality, multi_options):
        option_ids = multi_options.clamp(min=0, max=self.n_actions - 1)
        option_logits = clip_logits.gather(1, option_ids)
        option_probabilities = torch.sigmoid(option_logits)
        sorted_logits = torch.sort(option_logits, dim=1, descending=True).values
        sorted_probabilities = torch.sort(option_probabilities, dim=1, descending=True).values
        entropy = -(option_probabilities * torch.log(option_probabilities.clamp_min(1e-07)) + (1.0 - option_probabilities) * torch.log((1.0 - option_probabilities).clamp_min(1e-07))).mean(dim=1, keepdim=True)
        gaps = torch.cat([sorted_probabilities[:, 0:1] - sorted_probabilities[:, 1:2], sorted_probabilities[:, 1:2] - sorted_probabilities[:, 2:3], sorted_probabilities[:, 2:3] - sorted_probabilities[:, 3:4]], dim=1)
        summary = torch.cat([option_probabilities.sum(dim=1, keepdim=True), option_probabilities.mean(dim=1, keepdim=True), option_probabilities.max(dim=1, keepdim=True).values, option_probabilities.min(dim=1, keepdim=True).values, gaps, entropy], dim=1)
        action_context = self.action_option_embedding(option_ids).mean(dim=1)
        clip_context = self.cardinality_clip(pooled)
        features = torch.cat([sorted_logits, sorted_probabilities, summary, action_context, clip_context, modality], dim=1)
        return self.cardinality_head(features)

    def sequence_outputs(self, frame_logits, sequence_options):
        option_ids = sequence_options.clamp(min=0, max=self.n_actions - 1)
        selected = frame_logits.gather(2, option_ids.unsqueeze(1).expand(-1, self.n_steps, -1)).transpose(1, 2)
        attention = torch.softmax(selected, dim=-1)
        time = torch.linspace(0.0, 1.0, self.n_steps, device=frame_logits.device)
        centers = (attention * time).sum(dim=-1)
        peaks = attention.max(dim=-1).values
        overlap = torch.einsum('bit,bjt->bij', attention, attention)
        embeddings = self.action_option_embedding(option_ids)
        center_left = centers.unsqueeze(2).expand(-1, -1, 4)
        center_right = centers.unsqueeze(1).expand(-1, 4, -1)
        peak_left = peaks.unsqueeze(2).expand(-1, -1, 4)
        peak_right = peaks.unsqueeze(1).expand(-1, 4, -1)
        embedding_left = embeddings.unsqueeze(2).expand(-1, -1, 4, -1)
        embedding_right = embeddings.unsqueeze(1).expand(-1, 4, -1, -1)
        pair_features = torch.cat([center_left.unsqueeze(-1), center_right.unsqueeze(-1), (center_right - center_left).unsqueeze(-1), peak_left.unsqueeze(-1), peak_right.unsqueeze(-1), overlap.unsqueeze(-1), embedding_left, embedding_right], dim=-1)
        pair_logits = self.order_pair_head(pair_features).squeeze(-1)
        permutation_scores = []
        for permutation in PERMUTATIONS:
            score = torch.zeros(frame_logits.shape[0], device=frame_logits.device)
            for left_position, left in enumerate(permutation):
                for right in permutation[left_position + 1:]:
                    score = score + pair_logits[:, left, right]
            permutation_scores.append(score)
        return (torch.stack(permutation_scores, dim=1), pair_logits, centers)

    def forward(self, sensor, vision, motion, modality, source_id, multi_options, sequence_options, emotion_options, object_options):
        sensor_hidden = self.sensor_proj(self.sensor_norm(sensor))
        vision_hidden = self.vision_proj(self.vision_norm(vision))
        modality_steps = modality.unsqueeze(1).expand(-1, self.n_steps, -1)
        gate = torch.sigmoid(self.gate(torch.cat([sensor_hidden, vision_hidden, modality_steps], dim=-1)))
        hidden = gate * sensor_hidden + (1.0 - gate) * vision_hidden
        hidden = hidden + self.position + self.source_embedding(source_id).unsqueeze(1)
        hidden = self.temporal(hidden)
        frame_logits = self.action_head(hidden)
        clip_logits = torch.logsumexp(frame_logits, dim=1) - math.log(self.n_steps)
        pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1)
        motion_hidden = self.motion_encoder(motion)
        cardinality = self.cardinality_scores(clip_logits, pooled, modality, multi_options)
        sequence, pair_logits, sequence_centers = self.sequence_outputs(frame_logits, sequence_options)
        manner = torch.cat([motion_hidden, pooled, torch.sigmoid(clip_logits), modality], dim=1)
        emotion = self.emotion_option_scores(manner, emotion_options)
        object_scores = self.object_option_scores(pooled, object_options, self.object_embedding, self.object_clip, self.object_score)
        return {'clip_logits': clip_logits, 'frame_logits': frame_logits, 'cardinality': cardinality, 'sequence': sequence, 'pair_logits': pair_logits, 'sequence_centers': sequence_centers, 'emotion': emotion, 'object': object_scores}

def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

def batch_loss(model: MaskedMILV2, raw_batch: dict, device: torch.device, n_steps: int) -> tuple[torch.Tensor, dict[str, float]]:
    batch = move_batch(raw_batch, device)
    output = model(batch['sensor'], batch['vision'], batch['motion'], batch['modality'], batch['source_id'], batch['multi_options'], batch['sequence_options'], batch['emotion_options'], batch['object_options'])
    weighted_losses = {}
    clip_logits = output['clip_logits']
    frame_logits = output['frame_logits']
    if batch['known'].sum() > 0:
        raw = F.binary_cross_entropy_with_logits(clip_logits, batch['action'], reduction='none')
        weights = torch.where(batch['action'] > 0.5, torch.full_like(batch['action'], 2.4), torch.ones_like(batch['action']))
        weighted_losses['hau_action'] = (raw * batch['known'] * weights).sum() / (batch['known'] * weights).sum().clamp_min(1.0)
    harn_mask = batch['harn'] >= 0
    if harn_mask.any():
        weighted_losses['harn_action'] = 1.15 * F.cross_entropy(clip_logits[harn_mask], batch['harn'][harn_mask])
    if batch['order'].sum() > 0:
        time = torch.linspace(0.0, 1.0, n_steps, device=device)
        attention = torch.softmax(frame_logits.transpose(1, 2), dim=-1)
        centers = (attention * time).sum(dim=-1)
        pair_loss = F.softplus(centers.unsqueeze(2) - centers.unsqueeze(1) + 0.12)
        weighted_losses['order_pair'] = 0.25 * ((pair_loss * batch['order']).sum() / batch['order'].sum().clamp_min(1.0))
    multi_mask = batch['multi_count'] >= 1
    if multi_mask.any():
        weighted_losses['multi_count'] = 0.95 * F.cross_entropy(output['cardinality'][multi_mask], batch['multi_count'][multi_mask] - 1)
    sequence_mask = batch['sequence_target'] >= 0
    if sequence_mask.any():
        weighted_losses['sequence_listwise'] = 0.95 * F.cross_entropy(output['sequence'][sequence_mask], batch['sequence_target'][sequence_mask])
    emotion_mask = batch['emotion_target'] >= 0
    if emotion_mask.any():
        weighted_losses['emotion'] = 1.4 * F.cross_entropy(output['emotion'][emotion_mask], batch['emotion_target'][emotion_mask])
    object_mask = batch['object_target'] >= 0
    if object_mask.any():
        weighted_losses['object'] = F.cross_entropy(output['object'][object_mask], batch['object_target'][object_mask])
    if not weighted_losses:
        raise AssertionError('A training batch has no supervised objective')
    total = sum(weighted_losses.values())
    return (total, {name: float(value.detach().cpu()) for name, value in weighted_losses.items()})

@torch.inference_mode()
def infer_units(model: MaskedMILV2, units: list[str], targets: dict[str, dict], unit_index: dict[str, int], source_ids: np.ndarray, sensor: np.ndarray, vision: np.ndarray, motion: np.ndarray, modality: np.ndarray, device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    loader = DataLoader(UnitDataset(units, targets, unit_index, source_ids, sensor, vision, motion, modality), batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda', drop_last=False)
    predictions = {}
    model.eval()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch['sensor'], batch['vision'], batch['motion'], batch['modality'], batch['source_id'], batch['multi_options'], batch['sequence_options'], batch['emotion_options'], batch['object_options'])
        action_probability = torch.sigmoid(output['clip_logits'])
        for position, unit in enumerate(raw_batch['unit']):
            if unit in predictions:
                raise AssertionError(f'Duplicate inference unit {unit}')
            predictions[unit] = {'action': action_probability[position].cpu().numpy(), 'cardinality': output['cardinality'][position].cpu().numpy(), 'sequence': output['sequence'][position].cpu().numpy(), 'pair_logits': output['pair_logits'][position].cpu().numpy(), 'sequence_centers': output['sequence_centers'][position].cpu().numpy(), 'emotion': output['emotion'][position].cpu().numpy(), 'object': output['object'][position].cpu().numpy()}
    if set(predictions) != set(units) or len(predictions) != len(units):
        raise AssertionError('Inference did not return exactly the requested units')
    return predictions

def action_value(prediction: dict[str, np.ndarray], value: object, aliases: dict[str, str | tuple[str, ...]], class_index: dict[str, int]) -> float:
    resolved = aliases.get(norm_text(value))
    if isinstance(resolved, tuple) and len(resolved) > 1:
        probabilities = [float(prediction['action'][class_index[semantic]]) for semantic in resolved if semantic in class_index]
        if probabilities:
            return float(1.0 - np.prod(1.0 - np.clip(probabilities, 0.0, 1.0)))
    semantic = canonical_action(value, aliases)
    index = class_index.get(semantic)
    return float(prediction['action'][index]) if index is not None else 1e-06

def option_action_scores(row: pd.Series, prediction: dict[str, np.ndarray], aliases: dict[str, str | tuple[str, ...]], class_index: dict[str, int]) -> list[float]:
    category = str(row['category'])
    if category == 'combination':
        values = []
        for letter in available_letters(row):
            atom_scores = [action_value(prediction, atom, aliases, class_index) for atom in split_atoms(row[letter])]
            values.append(float(np.mean(atom_scores)) if atom_scores else 0.0)
        return values
    if category in {'single', 'multi', 'sequence'}:
        return [action_value(prediction, row[letter], aliases, class_index) for letter in available_letters(row)]
    return []

def decode_row_v2(row: pd.Series, prediction: dict[str, np.ndarray], aliases: dict[str, str | tuple[str, ...]], class_index: dict[str, int]) -> str:
    """Exact ambiguity-aware masked-MIL-v2 hard decoder."""
    letters = available_letters(row)
    category = str(row['category'])
    if category == 'single':
        scores = option_action_scores(row, prediction, aliases, class_index)
        return letters[int(np.argmax(scores))]
    if category == 'multi':
        probabilities = np.asarray(option_action_scores(row, prediction, aliases, class_index))
        count = int(np.argmax(prediction['cardinality'])) + 1
        chosen = sorted(np.argsort(-probabilities, kind='stable')[:count].tolist())
        return ''.join((letters[index] for index in chosen))
    if category == 'combination':
        scores = []
        for letter in letters:
            values = [action_value(prediction, atom, aliases, class_index) for atom in split_atoms(row[letter])]
            scores.append(sum((math.log(max(value, 1e-06)) for value in values)) / max(1, len(values)))
        return letters[int(np.argmax(scores))]
    if category == 'sequence':
        permutation = PERMUTATIONS[int(np.argmax(prediction['sequence']))]
        return ''.join((letters[index] for index in permutation))
    if category == 'emotion':
        return letters[int(np.argmax(prediction['emotion']))]
    if category == 'object_interaction':
        return letters[int(np.argmax(prediction['object']))]
    raise AssertionError(f'Unsupported category {category!r}')

def model_option_probabilities(row: pd.Series, prediction: dict[str, np.ndarray], aliases: dict[str, str | tuple[str, ...]], class_index: dict[str, int]) -> dict[str, float]:
    letters = available_letters(row)
    category = str(row['category'])
    if category in {'single', 'multi', 'sequence', 'combination'}:
        scores = option_action_scores(row, prediction, aliases, class_index)
    elif category == 'emotion':
        scores = prediction['emotion'][:len(letters)]
    elif category == 'object_interaction':
        scores = prediction['object'][:len(letters)]
    else:
        raise AssertionError(f'Unsupported category {category!r}')
    scores = np.asarray(scores, dtype=np.float64)
    if category in {'emotion', 'object_interaction'}:
        scores = np.exp(scores - scores.max())
    scores = np.clip(scores, 1e-12, None)
    scores /= scores.sum()
    return {letter: float(scores[index]) for index, letter in enumerate(letters)}

def anchored_raw_distribution(row: pd.Series, raw_prediction: str) -> dict[str, float]:
    candidates = candidate_values(row)
    if raw_prediction not in candidates:
        raise AssertionError(f"Raw masked-MIL-v2 prediction is illegal for {row['qa_id']}")
    if len(candidates) == 1:
        return {raw_prediction: 1.0}
    remainder = (1.0 - RAW_ANCHOR_PRIOR) / (len(candidates) - 1)
    distribution = {candidate: RAW_ANCHOR_PRIOR if candidate == raw_prediction else remainder for candidate in candidates}
    if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12):
        raise AssertionError('Graph-v2 raw-anchor distribution is not normalized')
    return distribution

def distribution_stats(distribution: dict[str, float]) -> dict[str, float | str]:
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    best, confidence = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    entropy = -sum((probability * math.log(max(probability, 1e-12)) for probability in distribution.values()))
    return {'prediction': best, 'confidence': float(confidence), 'margin': float(confidence - runner_up), 'entropy': float(entropy)}
GRAPH_ALIASES = {**ALIASES, 'drink': 'drinking water', 'listen to music with headphones': 'listening to music with headphones'}

def graph_norm_text(value: object) -> str:
    text = str(value or '').strip().lower().replace('&', ' and ')
    text = text.replace('oneself', 'self').replace('a mobile phone', 'phone')
    text = re.sub('\\b(the|a|an|their|his|her)\\b', ' ', text)
    text = re.sub('[^a-z0-9]+', ' ', text)
    return re.sub('\\s+', ' ', text).strip()
GRAPH_NORMALIZED_ALIASES = {graph_norm_text(key): graph_norm_text(value) for key, value in GRAPH_ALIASES.items()}

def graph_canonical_action(value: object) -> str:
    normalized = graph_norm_text(value)
    return GRAPH_NORMALIZED_ALIASES.get(normalized, normalized)

def candidate_actions(row: pd.Series, candidate: str, aliases: dict[str, str]) -> tuple[frozenset[str], tuple[str, ...]]:
    category = str(row['category'])
    if category == 'combination':
        actions = tuple((graph_canonical_action(atom) for atom in split_atoms(row[candidate])))
    else:
        actions = tuple((graph_canonical_action(row[letter]) for letter in candidate))
    actions = tuple((action for action in actions if action))
    if category == 'sequence':
        return (frozenset(actions), actions)
    return (frozenset(actions), tuple(sorted(actions)))

def clip_rows(frame: pd.DataFrame) -> dict[str, pd.Series]:
    rows = {}
    for category, part in frame.groupby('category', sort=False):
        if len(part) != 1:
            raise AssertionError(f"Clip {part.iloc[0]['unit']} has {len(part)} rows for category {category}")
        rows[str(category)] = part.iloc[0]
    return rows

def exact_rule_audit(fit_rows: pd.DataFrame, aliases: dict[str, str]) -> tuple[list[dict], dict[str, bool]]:
    counts = defaultdict(lambda: [0, 0])
    hau = fit_rows[fit_rows['source'].eq('HAU')]
    for _, group in hau.groupby('unit', sort=False):
        rows = clip_rows(group)
        single = rows.get('single')
        multi = rows.get('multi')
        combination = rows.get('combination')
        sequence = rows.get('sequence')
        if sequence is None:
            continue
        sequence_options = {graph_canonical_action(sequence[letter]) for letter in available_letters(sequence) if graph_canonical_action(sequence[letter])}
        if single is not None:
            matches = [letter for letter in available_letters(single) if graph_canonical_action(single[letter]) in sequence_options]
            if len(matches) == 1:
                answer = clean_prediction(single['answer'], single)
                counts['single_unique_sequence_overlap'][0] += 1
                counts['single_unique_sequence_overlap'][1] += int(matches[0] == answer)
        if combination is not None:
            matches = []
            for letter in available_letters(combination):
                option_actions, _ = candidate_actions(combination, letter, aliases)
                if option_actions and option_actions <= sequence_options:
                    matches.append(letter)
            if len(matches) == 1:
                answer = clean_prediction(combination['answer'], combination)
                counts['combination_unique_sequence_subset'][0] += 1
                counts['combination_unique_sequence_subset'][1] += int(matches[0] == answer)
        if multi is not None:
            answer = clean_prediction(multi['answer'], multi)
            if answer:
                for letter in available_letters(multi):
                    if graph_canonical_action(multi[letter]) in sequence_options:
                        counts['multi_sequence_forced_inclusion'][0] += 1
                        counts['multi_sequence_forced_inclusion'][1] += int(letter in answer)
    audit = []
    enabled = {}
    for rule in ('single_unique_sequence_overlap', 'combination_unique_sequence_subset', 'multi_sequence_forced_inclusion'):
        support, correct = counts[rule]
        accuracy = correct / support if support else 0.0
        is_enabled = support >= MIN_RULE_SUPPORT and accuracy >= MIN_RULE_ACCURACY
        audit.append({'rule': rule, 'support': int(support), 'correct': int(correct), 'accuracy': float(accuracy), 'minimum_support': MIN_RULE_SUPPORT, 'minimum_accuracy': MIN_RULE_ACCURACY, 'enabled': bool(is_enabled)})
        enabled[rule] = bool(is_enabled)
    return (audit, enabled)

def hard_constraints(rows: dict[str, pd.Series], enabled: dict[str, bool], aliases: dict[str, str]) -> dict:
    forced = {'single': None, 'combination': None, 'multi_include': set()}
    sequence = rows.get('sequence')
    if sequence is None:
        return forced
    sequence_options = {graph_canonical_action(sequence[letter]) for letter in available_letters(sequence) if graph_canonical_action(sequence[letter])}
    single = rows.get('single')
    if enabled.get('single_unique_sequence_overlap') and single is not None:
        matches = [letter for letter in available_letters(single) if graph_canonical_action(single[letter]) in sequence_options]
        if len(matches) == 1:
            forced['single'] = matches[0]
    combination = rows.get('combination')
    if enabled.get('combination_unique_sequence_subset') and combination is not None:
        matches = []
        for letter in available_letters(combination):
            option_actions, _ = candidate_actions(combination, letter, aliases)
            if option_actions and option_actions <= sequence_options:
                matches.append(letter)
        if len(matches) == 1:
            forced['combination'] = matches[0]
    multi = rows.get('multi')
    if enabled.get('multi_sequence_forced_inclusion') and multi is not None:
        forced['multi_include'] = {letter for letter in available_letters(multi) if graph_canonical_action(multi[letter]) in sequence_options}
    return forced

def allowed_candidates(row: pd.Series, forced: dict) -> list[str]:
    category = str(row['category'])
    candidates = candidate_values(row)
    if category == 'single' and forced.get('single'):
        candidates = [candidate for candidate in candidates if candidate == forced['single']]
    elif category == 'combination' and forced.get('combination'):
        candidates = [candidate for candidate in candidates if candidate == forced['combination']]
    elif category == 'multi' and forced.get('multi_include'):
        required = set(forced['multi_include'])
        candidates = [candidate for candidate in candidates if required <= set(candidate)]
    return candidates or candidate_values(row)

def apply_hard_stage(rows: dict[str, pd.Series], raw_predictions: dict[str, str], enabled: dict[str, bool], aliases: dict[str, str]) -> tuple[dict[str, str], list[dict]]:
    output = dict(raw_predictions)
    changes = []
    forced = hard_constraints(rows, enabled, aliases)
    for category in ('single', 'combination'):
        row = rows.get(category)
        candidate = forced.get(category)
        if row is None or not candidate:
            continue
        qa_id = str(row['qa_id'])
        before = output[qa_id]
        output[qa_id] = candidate
        if before != candidate:
            changes.append({'qa_id': qa_id, 'unit': str(row['unit']), 'category': category, 'stage': 'hard_exact', 'rule': f'forced_{category}', 'from_prediction': before, 'to_prediction': candidate})
    multi = rows.get('multi')
    required = set(forced.get('multi_include', set()))
    if multi is not None and required:
        qa_id = str(multi['qa_id'])
        before = output[qa_id]
        after = ''.join((letter for letter in available_letters(multi) if letter in set(before) | required))
        if after in set(candidate_values(multi)):
            output[qa_id] = after
            if before != after:
                changes.append({'qa_id': qa_id, 'unit': str(multi['unit']), 'category': 'multi', 'stage': 'hard_exact', 'rule': 'forced_multi_inclusion', 'from_prediction': before, 'to_prediction': after})
    return (output, changes)

def relation_signature(left_category: str, left_row: pd.Series, left_candidate: str, right_category: str, right_row: pd.Series, right_candidate: str, aliases: dict[str, str]) -> tuple:
    left_set, left_order = candidate_actions(left_row, left_candidate, aliases)
    right_set, right_order = candidate_actions(right_row, right_candidate, aliases)
    intersection = left_set & right_set
    signature = [len(left_set), len(right_set), len(intersection), int(bool(left_set) and left_set <= right_set), int(bool(right_set) and right_set <= left_set), int(bool(left_set) and left_set == right_set)]
    if left_category == 'sequence':
        positions = tuple(sorted((left_order.index(action) for action in right_set if action in left_order)))
        signature.extend((len(positions), positions))
    elif right_category == 'sequence':
        positions = tuple(sorted((right_order.index(action) for action in left_set if action in right_order)))
        signature.extend((len(positions), positions))
    else:
        signature.extend((0, ()))
    return tuple(signature)

def learn_graph_model(fit_rows: pd.DataFrame, aliases: dict[str, str]) -> dict:
    relation_counts = defaultdict(Counter)
    relation_totals = Counter()
    multi_cardinality = Counter()
    sequence_before = defaultdict(lambda: [0, 0])
    hau = fit_rows[fit_rows['source'].eq('HAU')]
    for _, row in hau[hau['category'].eq('multi')].iterrows():
        answer = clean_prediction(row['answer'], row)
        if answer:
            multi_cardinality[len(answer)] += 1
    for _, row in hau[hau['category'].eq('sequence')].iterrows():
        answer = clean_prediction(row['answer'], row)
        if not answer:
            continue
        _, ordered = candidate_actions(row, answer, aliases)
        for left_index, left_action in enumerate(ordered):
            for right_action in ordered[left_index + 1:]:
                sequence_before[left_action, right_action][0] += 1
                sequence_before[left_action, right_action][1] += 1
                sequence_before[right_action, left_action][1] += 1
    for _, group in hau.groupby('unit', sort=False):
        rows = clip_rows(group)
        categories = [category for category in TARGET_CATEGORIES if category in rows]
        for left_position, left_category in enumerate(categories):
            left_row = rows[left_category]
            left_answer = clean_prediction(left_row['answer'], left_row)
            if not left_answer:
                continue
            for right_category in categories[left_position + 1:]:
                right_row = rows[right_category]
                right_answer = clean_prediction(right_row['answer'], right_row)
                if not right_answer:
                    continue
                pair = (left_category, right_category)
                signature = relation_signature(left_category, left_row, left_answer, right_category, right_row, right_answer, aliases)
                relation_counts[pair][signature] += 1
                relation_totals[pair] += 1
    return {'relation_counts': relation_counts, 'relation_totals': relation_totals, 'multi_cardinality': multi_cardinality, 'sequence_before': sequence_before}

def relation_log_probability(model: dict, pair: tuple, signature: tuple) -> float:
    counts = model['relation_counts'].get(pair, Counter())
    total = model['relation_totals'].get(pair, 0)
    outcomes = len(counts) + 1
    numerator = counts.get(signature, 0) + LAPLACE_ALPHA
    denominator = total + LAPLACE_ALPHA * outcomes
    return math.log(numerator / max(denominator, 1e-12))

def sequence_log_probability(model: dict, row: pd.Series, candidate: str, aliases: dict[str, str]) -> float:
    _, ordered = candidate_actions(row, candidate, aliases)
    score = 0.0
    for left_index, left_action in enumerate(ordered):
        for right_action in ordered[left_index + 1:]:
            wins, total = model['sequence_before'].get((left_action, right_action), [0, 0])
            score += math.log((wins + 1.0) / (total + 2.0))
    return score

def category_prior_log_probability(model: dict, row: pd.Series, candidate: str, aliases: dict[str, str]) -> float:
    category = str(row['category'])
    if category == 'multi':
        counts = model['multi_cardinality']
        total = sum(counts.values())
        return math.log((counts[len(candidate)] + LAPLACE_ALPHA) / (total + 4.0 * LAPLACE_ALPHA))
    if category == 'sequence':
        return sequence_log_probability(model, row, candidate, aliases)
    return 0.0

def raw_log_probability(row: pd.Series, candidate: str, raw_prediction: str) -> float:
    candidates = candidate_values(row)
    if raw_prediction not in candidates or candidate not in candidates:
        raise AssertionError(f"Invalid graph-v2 candidate support for {row['qa_id']}")
    if candidate == raw_prediction:
        return math.log(RAW_ANCHOR_PRIOR)
    alternatives = max(1, len(candidates) - 1)
    probability = (1.0 - RAW_ANCHOR_PRIOR) / alternatives
    return math.log(max(probability, 1e-12))

def logsumexp_values(values: list[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(sum((math.exp(value - maximum) for value in values)))

def graph_decode(rows: dict[str, pd.Series], raw_predictions: dict[str, str], hard_predictions: dict[str, str], model: dict, enabled: dict[str, bool], aliases: dict[str, str]) -> tuple[dict[str, str], dict[str, dict[str, float]], dict]:
    categories = [category for category in TARGET_CATEGORIES if category in rows]
    output = dict(hard_predictions)
    posterior = {str(row['qa_id']): anchored_raw_distribution(row, raw_predictions[str(row['qa_id'])]) for row in rows.values()}
    if not categories:
        return (output, posterior, {'combinations': 0, 'map_score': 0.0, 'log_partition': 0.0})
    forced = hard_constraints(rows, enabled, aliases)
    candidate_lists = [allowed_candidates(rows[category], forced) for category in categories]
    combination_count = int(np.prod([len(values) for values in candidate_lists]))
    if combination_count > 5760:
        raise AssertionError(f'Clip graph has {combination_count} assignments; expected at most 5760')
    unary_scores = {}
    for category, candidates in zip(categories, candidate_lists):
        row = rows[category]
        qa_id = str(row['qa_id'])
        unary_scores[category] = {candidate: raw_log_probability(row, candidate, raw_predictions[qa_id]) + CATEGORY_PRIOR_WEIGHT * category_prior_log_probability(model, row, candidate, aliases) for candidate in candidates}
    pairwise_scores = {}
    for left_position, left_category in enumerate(categories):
        for right_position in range(left_position + 1, len(categories)):
            right_category = categories[right_position]
            pair = (left_category, right_category)
            if not model['relation_totals'].get(pair, 0):
                continue
            table = {}
            for left_candidate in candidate_lists[left_position]:
                for right_candidate in candidate_lists[right_position]:
                    signature = relation_signature(left_category, rows[left_category], left_candidate, right_category, rows[right_category], right_candidate, aliases)
                    table[left_candidate, right_candidate] = PAIR_FACTOR_WEIGHT * relation_log_probability(model, pair, signature)
            pairwise_scores[pair] = table
    assignments = []
    for values in itertools.product(*candidate_lists):
        assignment = dict(zip(categories, values))
        score = sum((unary_scores[category][candidate] for category, candidate in assignment.items()))
        for (left_category, right_category), table in pairwise_scores.items():
            score += table[assignment[left_category], assignment[right_category]]
        assignments.append((tuple(values), float(score)))
    if len(assignments) != combination_count:
        raise AssertionError('Graph assignment enumeration is incomplete')
    best_values, best_score = min(assignments, key=lambda item: (-item[1], item[0]))
    for category, candidate in zip(categories, best_values):
        output[str(rows[category]['qa_id'])] = candidate
    log_partition = logsumexp_values([score for _, score in assignments])
    for category_position, category in enumerate(categories):
        candidate_log_mass = {}
        for candidate in candidate_lists[category_position]:
            candidate_log_mass[candidate] = logsumexp_values([score for values, score in assignments if values[category_position] == candidate])
        restricted = {candidate: math.exp(value - log_partition) for candidate, value in candidate_log_mass.items()}
        all_candidates = candidate_values(rows[category])
        posterior[str(rows[category]['qa_id'])] = {candidate: float(restricted.get(candidate, 0.0)) for candidate in all_candidates}
    return (output, posterior, {'combinations': combination_count, 'map_score': float(best_score), 'log_partition': float(log_partition), 'decoded_categories': categories})
