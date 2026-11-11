"""Original Large MIL training from official raw inputs and labeled training QA."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source import masked_mil,masked_mil_core as core,sensor_preprocessing,siglip2_visual

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,list):return [plain(v) for v in value]
    if isinstance(value,tuple):return tuple(plain(v) for v in value)
    if isinstance(value,np.generic):return value.item()
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['training-questions','test-questions','sensors','test-videos','vision-model','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--train-videos',type=Path,action='append',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads);core.seed_everything(core.SEED);started=time.monotonic()
    train=pd.read_csv(args.training_questions,dtype=str).fillna('')
    test_all=pd.read_csv(args.test_questions,dtype=str).fillna('')
    if any(test_all[k].astype(str).str.strip().ne('').any() for k in ('answer','prediction','label','target') if k in test_all):
        raise ValueError('Test questions contain answers')
    public=['qa_id','source','path','category','question','A','B','C','D']
    test=test_all[public].copy();train,test=core.validate_input_tables(train,test)
    sensor_dir=out/'sensor_features';sensor_preprocessing.extract(args.sensors,sensor_dir,kind='base')
    vision_dir=out/'video_features'
    inputs=siglip2_visual.plan(args.train_videos,args.test_videos)
    siglip2_visual.extract(inputs,vision_dir,args.device,vision_model_directory=args.vision_model)
    paths={'sensor_features.npz':sensor_dir/'full_features.npz','sensor_units.json':sensor_dir/'full_units.json',
           'sensor_feature_manifest.json':sensor_dir/'sensor_feature_manifest.json'}
    paths.update({n:vision_dir/n for n in ['siglip_video_features.npz','siglip_video_units.json','siglip_feature_manifest.json']})
    core.find_input_file=lambda name,fragment:paths[name]
    features=core.load_and_align_features(train,test)
    model,ontology,aliases=masked_mil.initialize_model(train,features)
    targets,audit=core.build_targets(train,test,features['all_units'],features['train_units'],ontology,aliases)
    history=masked_mil.fit_model(model,targets,features,device=torch.device(args.device),progress=lambda r:print(json.dumps(r),flush=True))
    history.to_csv(out/'training_history.csv',index=False)
    positions=np.asarray([features['unit_index'][u] for u in features['train_units']],np.int64)
    positions=positions[features['modality'][positions,0]>.5]
    sensor=features['sensor'][positions]
    with np.load(sensor_dir/'full_features.npz',allow_pickle=False) as a:global_features=a['global_features'].astype(np.float32)
    sensor_units=json.loads((sensor_dir/'full_units.json').read_text());index={u:i for i,u in enumerate(sensor_units)}
    globals_selected=global_features[[index[features['all_units'][i]] for i in positions]]
    delta=np.abs(np.diff(sensor,axis=1))
    motion=np.concatenate([globals_selected,sensor.std(1),delta.mean(1),delta.std(1),delta.max(1)],axis=1).astype(np.float32)
    mean=motion.mean(0,dtype=np.float64).astype(np.float32);std=motion.std(0,dtype=np.float64).astype(np.float32)
    std=np.where(std>1e-5,std,1.).astype(np.float32)
    _,enabled=core.exact_rule_audit(train,{})
    checkpoint=dict(format='cuhkx-large-mil-deployment',state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        ontology=ontology,aliases=aliases,motion_mean=torch.from_numpy(mean),motion_std=torch.from_numpy(std),
        enabled_rules=enabled,graph_model=plain(core.learn_graph_model(train,{})))
    torch.save(checkpoint,out/'model.pth')
    report=dict(epochs=len(history),training_rows=len(train),training_recordings=len(features['train_units']),
        target_audit=audit,checkpoint_sha256=sha(out/'model.pth'),training_qa_sha256=sha(args.training_questions),
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(masked_mil.__file__),Path(core.__file__)]},
        historical_prediction_inputs=[],elapsed_seconds=time.monotonic()-started)
    (out/'training_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
