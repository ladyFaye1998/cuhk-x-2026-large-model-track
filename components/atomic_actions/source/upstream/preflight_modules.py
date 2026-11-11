"""Import every production stage in its declared runtime before model fitting."""
from pathlib import Path
import argparse
import importlib
import json
import sys

HERE=Path(__file__).resolve().parent
MODULES={
 'upstream_cpu': {'upstream':['native_stage','sensor_stage','kinematic_stage'],
                 'sensor_prepare':['prepare_sensor','sensor_original'],
                 'observations':['official_skeleton'],
                 'raw':['raw_adaptive_prepare','raw_adaptive_stage'],
                 'index_source':['official_archive_transport','generate_official_indexes']},
 'upstream': {'skate_fit':['skateformer_fit'], 'upstream':['skateformer_stage'],
              'efficient_fit':['efficientgcn_train'], 'raw':['raw_adaptive_stage']},
 'downstream': {'downstream':['regenerate_pipeline','run_pipeline','teacher_training','student_training','infer']}
}

def run(group):
    checked=[]
    for folder,names in MODULES[group].items():
        sys.path.insert(0,str(HERE/folder))
        for name in names:
            module=importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(HERE/folder):
                raise RuntimeError('A production module resolved outside the copied source bundle')
            checked.append(folder+'/'+name+'.py')
        sys.path.pop(0)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Import preflight unexpectedly initialized the accelerator')
    return {'runtime_group':group,'imported_sources':checked,'training_executed':False,'cuda_initialized':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=MODULES,required=True)
    print(json.dumps(run(p.parse_args().group)))
