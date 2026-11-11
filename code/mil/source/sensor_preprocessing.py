"""Regenerate Large Track sensor features directly from raw files or an archive."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import zipfile

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Observations:
    def __init__(self, raw):
        self.raw = Path(raw).resolve()
        self.archive = zipfile.ZipFile(self.raw) if self.raw.is_file() else None
        if not self.archive and not self.raw.is_dir():
            raise FileNotFoundError(self.raw)
        self.hashes, self.failed = {}, []

    def files(self):
        if self.archive:
            return sorted((r for r in self.archive.infolist() if not r.is_dir()
                           and Path(r.filename).suffix.lower() in ('.csv', '.json')), key=lambda r:r.filename)
        if shutil.which('rg'):
            paths = subprocess.run(['rg', '--files', str(self.raw), '-g', '*.csv', '-g', '*.json'],
                                   check=True, capture_output=True, text=True, encoding='utf-8').stdout.splitlines()
        else:
            paths = [str(Path(root)/name) for root, _, names in os.walk(self.raw)
                     for name in names if Path(name).suffix.lower() in ('.csv', '.json')]
        return [SimpleNamespace(filename=str(Path(p)).replace('\\','/')) for p in sorted(paths)]

    def read(self, info):
        name = info.filename
        try:
            data = self.archive.read(info) if self.archive else Path(name).read_bytes()
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            self.failed.append(dict(path=name, error=type(error).__name__))
            raise
        key = name if self.archive else Path(name).relative_to(self.raw).as_posix()
        self.hashes[key] = hashlib.sha256(data).hexdigest()
        return data

    def close(self):
        if self.archive:
            self.archive.close()


def extract(raw, output, kind='base', probe=False):
    if kind == 'base':
        from . import base_sensor_features as parser
    elif kind == 'rich':
        from . import sensor_features as parser
    else:
        raise ValueError('Expected base or rich sensors')
    output = Path(output)
    prefix = 'probe' if probe else 'full'
    if (output/(prefix+'_features.npz')).exists():
        raise FileExistsError('Use a fresh sensor output directory')
    started = time.monotonic()
    reader = Observations(raw)
    try:
        files = reader.files()
        grouped = defaultdict(lambda:defaultdict(list))
        for item in files:
            name = '/'+item.filename.replace('\\','/').lstrip('/')
            key, mode = parser.unit_key(name), parser.modality(name)
            if key and mode:
                grouped[key][mode].append(item)
        units = sorted(grouped)
        if len(units) != 3932:
            raise ValueError(f'Expected 3932 raw sensor units, found {len(units)}')
        chosen = units
        if probe:
            chosen = []
            for category in ('HAU/', 'HARn/', 'large_model_track_test/'):
                candidates = [u for u in units if u.startswith(category)]
                if not candidates:
                    raise ValueError('Missing raw source: '+category)
                chosen.extend(candidates[i] for i in (0, len(candidates)//2, len(candidates)-1))
        segments = np.zeros((len(chosen), parser.N_BINS, parser.SEGMENT_WIDTH), np.float32)
        masks = np.zeros((len(chosen),3), np.uint8)
        for index, key in enumerate(chosen):
            modes = grouped[key]
            segments[index] = np.concatenate([
                parser.parse_imu(reader, modes.get('imu', [])),
                parser.parse_skeleton(reader, modes.get('skeleton', [])),
                parser.parse_radar(reader, modes.get('radar', []))], axis=1)
            masks[index] = [bool(modes.get(m)) for m in ('imu','radar','skeleton')]
            if (index+1)%100 == 0 or probe:
                print(json.dumps(dict(processed=index+1, units=len(chosen),
                                      elapsed_seconds=time.monotonic()-started)), flush=True)
        if reader.failed:
            raise IOError('Raw observation reads failed: '+str(reader.failed[:5]))
        parts = [segments.mean(1),segments.std(1),segments.min(1),segments.max(1)]
        if kind == 'rich':
            delta = np.abs(np.diff(segments,axis=1))
            parts.extend([delta.mean(1),delta.max(1)])
        global_features = np.concatenate(parts+[masks.astype(np.float32)], axis=1).astype(np.float32)
        if not np.isfinite(segments).all() or not np.isfinite(global_features).all():
            raise ValueError('Nonfinite raw sensor features')
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/(prefix+'_features.npz'), segments=segments,
                            global_features=global_features, modality_mask=masks)
        (output/(prefix+'_units.json')).write_text(json.dumps(chosen,indent=2)+'\n')
        (output/(prefix+'_input_hashes.json')).write_text(json.dumps(reader.hashes,sort_keys=True,indent=2)+'\n')
        report = dict(status='completed', kind=kind, probe=probe, processed_units=len(chosen),
                      discovered_units=len(units), source_files=len(files), read_files=len(reader.hashes),
                      failed_raw_reads=reader.failed, elapsed_seconds=time.monotonic()-started,
                      source_sha256={p.name:sha(p) for p in (Path(__file__),Path(parser.__file__))},
                      raw_source=str(reader.raw), input_kind='raw observation archive' if reader.archive else 'raw observation directory',
                      historical_feature_or_prediction_inputs=False,
                      artifacts={prefix+suffix:sha(output/(prefix+suffix)) for suffix in
                                 ('_features.npz','_units.json','_input_hashes.json')})
        (output/(prefix+'_validation.json')).write_text(json.dumps(report,indent=2)+'\n')
        if kind == 'base' and not probe:
            from .base_sensor_stage import materialize
            materialize(output)
        return report
    finally:
        reader.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kind', choices=('base','rich'), default='base')
    ap.add_argument('--probe', action='store_true')
    args = ap.parse_args()
    print(json.dumps(extract(args.raw,args.output,args.kind,args.probe),indent=2))
