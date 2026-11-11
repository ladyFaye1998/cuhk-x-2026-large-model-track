"""Regenerate original IMU/radar/skeleton descriptors from official archive bytes."""
import argparse
import hashlib
import importlib.util
import json
import threading
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fact(path, role):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'role': role}

def prepare(index_root, output):
    recipe = load('sensor_original', ROOT / 'sensor_original.py')
    transport = load('sensor_transport', ROOT / 'sensor_transport.py')
    # This mapping is independently recorded by the verified official IMU reader.
    transport.FILES['1QbsW4TupA2QMuE8ckph-YT9kP_MKRakV'] = (5368709120, 'Small-Model-Track/Training/data/HAR.z03')
    index_root, output = Path(index_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    recipe.INPUT_ROOT, recipe.OUTPUT_ROOT = index_root, output
    train_index = index_root / 'nonvisual_entries.jsonl.gz'
    test_index = index_root / 'test_nonvisual_entries.jsonl.gz'
    train = [e for e in recipe.read_index(train_index) if recipe.is_useful(e)]
    test = [e for e in recipe.read_index(test_index) if recipe.is_useful(e)]
    train_plan = []
    for disk in sorted({e['disk'] for e in train}):
        rows = [e for e in train if e['disk'] == disk]
        start = min(e['offset'] for e in rows)
        end = max(e['offset'] + e['compressed'] + 1024 for e in rows)-1
        train_plan.append({'disk': disk, 'file_id': recipe.TRAIN_DISK_IDS[disk], 'start': start, 'end': end, 'bytes': end-start+1})
    test_ranges = recipe.merged_intervals(test)
    budget = sum(r['bytes'] for r in train_plan)+sum(end-start+1 for start,end,_ in test_ranges)
    if budget > 500_000_000:
        raise ValueError('Sensor preparation exceeds the planned 500 MB transfer bound')
    plan = {'train': train_plan, 'test_ranges': len(test_ranges), 'total_http_payload_bytes': budget,
            'train_members': len(train), 'test_members': len(test), 'gpu_used': False}
    (output / 'download_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan), flush=True)
    ranges, decoded = [], {'train': 0, 'test': 0}
    guard = threading.Lock()
    def fetch(file_id, start, end, attempts=8):
        blob = transport.fetch_range(file_id, start, end, attempts)
        with guard:
            ranges.append({'file_id': file_id, 'start': start, 'end': end, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()})
        return blob
    recipe.fetch_range = fetch
    # Validate member integrity in addition to the original decoder; return exact bytes.
    for cls, split in ((recipe.ContiguousStore, 'train'), (recipe.SparseStore, 'test')):
        original = cls.raw
        def checked(self, entry, original=original, split=split):
            blob = original(self, entry)
            if len(blob) != entry['uncompressed'] or zlib.crc32(blob) != entry['crc']:
                raise ValueError('Official member CRC/length mismatch')
            decoded[split] += 1
            return blob
        cls.raw = checked
    started = time.time()
    recipe.main()
    if sum(r['bytes'] for r in ranges) != budget:
        raise ValueError('Completed transfer differs from its fixed byte plan')
    artifacts = []
    for name in ('small_sensor_features.npz', 'train_sensor_metadata.csv', 'test_sensor_metadata.csv'):
        path = output / name
        if name.endswith('.csv'):
            # Platform-independent serialization; values and row order are unchanged.
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        artifacts.append(fact(path, 'fresh_official_observation_features_or_metadata'))
    with np.load(output / 'small_sensor_features.npz', allow_pickle=False) as data:
        tensor_hashes = {key: hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest() for key in data.files}
    report = {'status': 'completed', 'seconds': time.time()-started, 'learned_inputs_used': [],
              'inputs': [fact(train_index, 'official_archive_index'), fact(test_index, 'official_archive_index')],
              'source_files': [fact(ROOT / name, 'executable_source') for name in ('prepare_sensor.py', 'sensor_original.py', 'sensor_transport.py')],
              'range_plan': plan, 'ranges': ranges, 'decoded_CRC_checked_members': decoded,
              'outputs': artifacts, 'tensor_sha256': tensor_hashes, 'transport': transport.receipt(),
              'numerical_preprocessing_changed': False, 'gpu_used': False,
              'training_executed': False, 'prediction_or_reference_files_read': False}
    (output / 'preparation_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': 'completed', 'seconds': report['seconds'], 'downloaded_bytes': budget, 'decoded': decoded}), flush=True)
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.index_root, args.output)
