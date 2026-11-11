"""Bind freshly fitted fold checkpoints to their forward stage; reject partial fits."""
from pathlib import Path
import argparse
import hashlib
import json

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): result.update(block)
    return result.hexdigest()

def safe(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Checkpoint path escapes its fresh fitting root')
    return path

def bind_skate(fit_root,replay_root):
    fit_root,replay_root=Path(fit_root).resolve(),Path(replay_root).resolve()
    manifest_path=fit_root/'training_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    mapping=json.loads((fit_root/'checkpoint_map.json').read_text())
    if (manifest.get('all_six_folds_trained') is not True or sorted(manifest.get('folds_completed',[]))!=list(range(6))
        or mapping.get('ready_for_six_fold_inference') is not True or len(mapping.get('folds',[]))!=6):
        raise ValueError('Six complete fresh joint/bone fits are mandatory')
    if len(manifest.get('training_reports',[]))!=12:
        raise ValueError('All twelve freshly trained models need retained fitting reports')
    learned=manifest.get('learned_inputs_used',[])
    if len(learned)!=2 or any(r.get('role')!='public_NTU120_pretraining_initialization' for r in learned):
        raise ValueError('Only the public initialization pair may enter SkateFormer fitting')
    expected={}
    for fold in mapping['folds']:
        for modality in ('joint','bone'):
            record=fold[modality];path=safe(fit_root,record['path'])
            if sha(path)!=record['sha256']: raise ValueError('Fresh SkateFormer checkpoint hash differs')
            expected[str(path)]=record['sha256']
    replay_path=replay_root/'stage_manifest.json'
    replay=json.loads(replay_path.read_text())
    actual={str(Path(r['path']).resolve()):r['sha256'] for r in replay.get('learned_inputs_used',[])}
    if actual!=expected or len(actual)!=12: raise ValueError('Replay did not use exactly the twelve new fold states')
    replay['inputs'].append({'path':str(manifest_path),'sha256':sha(manifest_path),'role':'fresh_skateformer_fitting_manifest'})
    replay['source_files'].append({'path':str(Path(__file__).resolve()),'sha256':sha(__file__)})
    replay['method'].update(fine_tuning_executed=True,
        fine_tuning_scope='Twelve fixed-recipe transfer fits precede this forward stage in the same empty run directory',
        fold0='Freshly trained backbone-transfer fold zero with the same fixed public initialization recipe')
    replay['technical_scope']='Complete recorded fitting path and saved-model inference executed.'
    replay_path.write_text(json.dumps(replay,indent=2)+'\n')

def verify_all(stages):
    stages=Path(stages)
    efficient=json.loads((stages/'efficientgcn/stage_manifest.json').read_text())
    if efficient.get('stage')!='efficientgcn' or efficient.get('status')!='all_six_folds_fully_trained' or efficient.get('method',{}).get('fine_tuning_executed') is not True:
        raise ValueError('EfficientGCN must be fully fitted in this run')
    kinematic=json.loads((stages/'kinematic/stage_manifest.json').read_text())
    if kinematic.get('training_executed_in_this_run') is not True or kinematic.get('learned_inputs_used')!=[]:
        raise ValueError('Kinematic stage must use its fresh fitting path')
    sensor=json.loads((stages/'sensor/stage_manifest.json').read_text())
    if sensor.get('status')!='completed' or sensor.get('learned_inputs_used')!=[] or len(sensor.get('fold_results',[]))!=6:
        raise ValueError('Sensor stage must contain six fresh training folds')
    raw=json.loads((stages/'raw_adaptive/stage_manifest.json').read_text())
    if raw.get('status')!='completed' or raw.get('learned_inputs_used')!=[]:
        raise ValueError('Raw-adaptive stage must be freshly fitted without learned inputs')
    skate=json.loads((stages/'skateformer/stage_manifest.json').read_text())
    if skate.get('method',{}).get('fine_tuning_executed') is not True:
        raise ValueError('SkateFormer fitting/replay binding is absent')
    return {'fresh_fitting_stages':['sensor','kinematic','raw_adaptive','skateformer','efficientgcn'],
            'native':'disclosed initialization forward stage, no fresh pretraining claim'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bind-skate');b.add_argument('--fit-root',required=True);b.add_argument('--replay-root',required=True)
    v=sub.add_parser('verify-all');v.add_argument('--stages',required=True)
    a=p.parse_args()
    if a.command=='bind-skate':bind_skate(a.fit_root,a.replay_root)
    else:print(json.dumps(verify_all(a.stages)))
