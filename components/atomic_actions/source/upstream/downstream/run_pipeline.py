"""CPU training reconstruction and fresh checkpoint inference."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits
import raw_inputs
from teacher_training import fit_teachers
from student_training import fit_student,predict,sha

def write_csv(path,paths,prediction):
    assert len(paths)==len(prediction) and len(set(paths))==len(paths)
    assert np.all((prediction>=0)&(prediction<40))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(('path','prediction'))
        writer.writerows(zip(paths,map(int,prediction),strict=True))

def record_training_result(out, report, provenance_path=None):
    """Record classifier training artifacts for the enclosing Large package."""
    out = Path(out)
    (out / 'run_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if provenance_path is not None:
        shutil.copyfile(provenance_path, out / 'input_provenance.json')


def run(assets,template,out,checkpoint=None,observation_cache=None,provenance_path=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    if observation_cache:
        with np.load(observation_cache,allow_pickle=False) as z:sequences=z['test'];counts=z['test_n_valid']
        paths=raw_inputs.read_sample_paths(Path(template));raw={'source':'local official observation cache'}
    else:sequences,counts,paths,raw=raw_inputs.download_test(Path(template))
    if checkpoint is None:
        assets=Path(assets)
        manifest=json.loads((assets/'asset_manifest.json').read_text())
        for name,fact in manifest.items():
            path=assets/name
            if path.stat().st_size!=fact['bytes'] or sha(path)!=fact['sha256']:raise RuntimeError('Input fingerprint mismatch: '+name)
        pseudo,teacher_report=fit_teachers(assets,out/'teacher_fit')
        checkpoint,student_report=fit_student(assets,out,sequences,counts,pseudo)
    else:
        checkpoint=Path(checkpoint);pseudo=None;teacher_report=None;student_report=None
    start=time.perf_counter();prediction,extraction=predict(checkpoint,sequences,counts);inference_seconds=time.perf_counter()-start
    repeated,_=predict(checkpoint,sequences,counts)
    if not np.array_equal(prediction,repeated):raise RuntimeError('CPU repeat mismatch')
    indices=np.linspace(0,len(prediction)-1,16).round().astype(np.int64)[::-1]
    indices=np.concatenate([indices,indices[:2]])
    selected,_=predict(checkpoint,sequences[indices],counts[indices])
    if not np.array_equal(selected,prediction[indices]):raise RuntimeError('Order or duplicate inference mismatch')
    if pseudo is not None and not np.array_equal(prediction,pseudo):raise RuntimeError('Reloaded model changed its pseudo-target fit')
    if checkpoint.stat().st_size>=100_000_000:raise RuntimeError('Inference checkpoint exceeds the size limit')
    write_csv(out/'submission.csv',paths,prediction)
    if os.name=='nt':
        import psutil
        peak=psutil.Process().memory_info().peak_wset/1024**2
    else:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report={'status':'downstream_training_and_checkpoint_inference_complete' if pseudo is not None else 'checkpoint_inference_complete',
        'rows':len(prediction),'csv_bytes':(out/'submission.csv').stat().st_size,'csv_sha256':sha(out/'submission.csv'),
        'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_sha256':sha(checkpoint),
        'cpu_inference_seconds':inference_seconds,'peak_process_ram_mib':peak,'gpu_memory_bytes':0,
        'cpu_repeat_exact':True,'order_and_duplicate_checks':len(indices),'total_seconds':time.perf_counter()-started,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':str(torch.__version__)},
        'raw_observations':raw,'feature_extraction':extraction,'teacher_training':teacher_report,'student_training':student_report,
        'scope':'Downstream fitting from verified regenerated upstream stage arrays; each upstream manifest records whether fitting or saved-state inference produced it.'}
    if provenance_path is not None:
        provenance_path=Path(provenance_path)
        provenance=json.loads(provenance_path.read_text())
        report['regenerated_input_provenance']={'file':'input_provenance.json',
            'sha256':sha(provenance_path),
            'verified_stage_count':len(provenance['upstream_stage_receipts'])}
    record_training_result(out,report,provenance_path)
    print(json.dumps({k:v for k,v in report.items() if k not in ['teacher_training','student_training','feature_extraction']},indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path);parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--observation-cache',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.assets,args.template,args.out,args.checkpoint,args.observation_cache)
