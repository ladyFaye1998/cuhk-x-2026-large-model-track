"""Bind regenerated raw sensor arrays to the MIL input interface."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def materialize(directory):
    directory=Path(directory)
    validation=json.loads((directory/'full_validation.json').read_text())
    units=json.loads((directory/'full_units.json').read_text())
    if validation['failed_raw_reads'] or validation['processed_units']!=len(units):
        raise ValueError('Raw sensor preprocessing did not complete')
    if len(units)!=3932 or len(set(units))!=len(units):raise ValueError('Unexpected sensor unit coverage')
    with np.load(directory/'full_features.npz',allow_pickle=False) as archive:
        segments=archive['segments'];global_features=archive['global_features'];mask=archive['modality_mask']
        if segments.shape!=(3932,8,263) or global_features.shape!=(3932,1055) or mask.shape!=(3932,3):
            raise ValueError('Regenerated sensor feature dimensions differ')
        if not np.isfinite(segments).all() or not np.isfinite(global_features).all():
            raise ValueError('Nonfinite regenerated sensor features')
        report=dict(unit_count=len(units),training_unit_count=sum(not k.startswith('large_model_track_test/') for k in units),
                    testing_unit_count=sum(k.startswith('large_model_track_test/') for k in units),bins=8,
                    segment_width=263,global_width=1055,segments_shape=list(segments.shape),
                    global_shape=list(global_features.shape),modality_coverage=mask.sum(axis=0).tolist(),
                    nonzero_fraction=float(np.count_nonzero(segments)/segments.size),errors=[],
                    origin='Regenerated raw IMU, radar and skeleton observations',
                    raw_preprocessing_seconds=validation['elapsed_seconds'])
    (directory/'sensor_feature_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    receipt=dict(status='completed',historical_prediction_inputs=False,original_manifest_inputs=False,
                 artifact_sha256={n:sha(directory/n) for n in ('full_features.npz','full_units.json','sensor_feature_manifest.json')},
                 raw_preprocessing_receipts={n:sha(directory/n) for n in ('full_validation.json','full_input_hashes.json')},
                 source_sha256=sha(__file__))
    (directory/'raw_sensor_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def verify(directory):
    directory=Path(directory)
    receipt=json.loads((directory/'raw_sensor_receipt.json').read_text())
    if receipt['status']!='completed' or receipt['historical_prediction_inputs'] or receipt['original_manifest_inputs']:
        raise ValueError('Expected regenerated raw sensor stage')
    for name,digest in receipt['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('Changed regenerated sensor artifact: '+name)
    for name,digest in receipt['raw_preprocessing_receipts'].items():
        if sha(directory/name)!=digest:raise ValueError('Changed raw preprocessing receipt: '+name)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path)
    print(json.dumps(materialize(parser.parse_args().directory),indent=2))
