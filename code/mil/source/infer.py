"""Large-track saved-model inference from raw videos and sensor observations."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source import masked_mil_core as core
from source import base_sensor_features as sensor_parser
from source import siglip2_visual
from source.sensor_preprocessing import Observations

PUBLIC_COLUMNS=['qa_id','source','path','category','question','A','B','C','D']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def questions(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not set(PUBLIC_COLUMNS)<=set(reader.fieldnames or []):raise ValueError('Missing question columns')
        rows=[]
        for row in reader:
            if any(row.get(k,'').strip() for k in ('answer','prediction','target','label')):
                raise ValueError('Inference question table contains answers')
            rows.append({k:row[k] or '' for k in PUBLIC_COLUMNS})
    frame=pd.DataFrame(rows)
    if frame.empty or frame.qa_id.duplicated().any():raise ValueError('Empty or duplicate question identifiers')
    frame['unit']=frame.path.map(core.test_unit_from_path)
    if frame.unit.isna().any():raise ValueError('Invalid raw recording path')
    if not frame.source.isin(['HAU','HARn']).all():raise ValueError('Unknown recording source')
    if frame.groupby('unit').source.nunique().max()!=1:raise ValueError('Conflicting source for one recording')
    return frame

def extract_sensors(raw,units,out):
    reader=Observations(raw)
    try:
        grouped=defaultdict(lambda:defaultdict(list));wanted=set(units)
        for item in reader.files():
            name='/'+item.filename.replace('\\','/').lstrip('/')
            key,mode=sensor_parser.unit_key(name),sensor_parser.modality(name)
            if key in wanted and mode:grouped[key][mode].append(item)
        segments=np.zeros((len(units),8,263),np.float32)
        masks=np.zeros((len(units),3),np.uint8)
        present=np.zeros(len(units),np.float32)
        for i,unit in enumerate(units):
            if unit not in grouped:continue
            modes=grouped[unit];present[i]=1
            segments[i]=np.concatenate([sensor_parser.parse_imu(reader,modes.get('imu',[])),
                sensor_parser.parse_skeleton(reader,modes.get('skeleton',[])),
                sensor_parser.parse_radar(reader,modes.get('radar',[]))],axis=1)
            masks[i]=[bool(modes.get(m)) for m in ('imu','radar','skeleton')]
            if (i+1)%40==0:print('Raw sensor recordings',i+1,'/',len(units),flush=True)
        if reader.failed:raise IOError('A raw sensor read failed')
        global_features=np.concatenate([segments.mean(1),segments.std(1),segments.min(1),segments.max(1),masks.astype(np.float32)],axis=1).astype(np.float32)
        if not np.isfinite(segments).all() or not np.isfinite(global_features).all():raise ValueError('Nonfinite sensor descriptor')
        np.savez_compressed(out/'sensor_observations.npz',segments=segments,global_features=global_features,modality_mask=masks,present=present)
        (out/'sensor_input_hashes.json').write_text(json.dumps(reader.hashes,sort_keys=True,indent=2)+'\n')
        return segments,global_features,present
    finally:reader.close()

def model_input(sensor,global_features,present,vision,state):
    delta=np.diff(sensor,axis=1)
    motion=np.concatenate([global_features,sensor.std(1),np.abs(delta).mean(1),np.abs(delta).std(1),np.abs(delta).max(1)],axis=1).astype(np.float32)
    motion=np.clip((motion-state['motion_mean'].numpy())/state['motion_std'].numpy(),-8.,8.).astype(np.float32)
    motion[present<.5]=0
    modality=np.stack([present,np.ones(len(present),np.float32)],axis=1)
    return dict(sensor=sensor,vision=vision.astype(np.float32),motion=motion,modality=modality)

def predict(frame,units,features,state):
    ontology=state['ontology'];weights=state['state_dict']
    model=core.MaskedMILV2(sensor_width=263,vision_width=1152,n_steps=8,n_actions=len(ontology['class_names']),
        motion_width=2107,emotion_semantics=weights['emotion_semantic.weight'].numpy(),
        emotion_vocab_size=len(ontology['emotion_vocab']),object_vocab_size=len(ontology['object_vocab']))
    model.load_state_dict(weights,strict=True);model.eval();core.NUM_WORKERS=0
    # Public options configure the question-specific scoring heads; there are no test targets.
    empty=pd.DataFrame(columns=['unit','source','category','path','answer','A','B','C','D','qa_id'])
    targets,_=core.build_targets(empty,frame,units,[],ontology,state['aliases'])
    sources=frame.groupby('unit').source.first().to_dict()
    source_ids=np.asarray([0 if sources[u]=='HAU' else 1 for u in units],np.int64)
    with torch.inference_mode():
        evidence=core.infer_units(model,units,targets,{u:i for i,u in enumerate(units)},source_ids,
            features['sensor'],features['vision'],features['motion'],features['modality'],torch.device('cpu'))
    raw={str(row.qa_id):core.decode_row_v2(row,evidence[row.unit],state['aliases'],ontology['class_index']) for _,row in frame.iterrows()}
    final=dict(raw)
    for _,group in frame.groupby('unit',sort=False):
        rows=core.clip_rows(group)
        group_raw={str(row['qa_id']):raw[str(row['qa_id'])] for row in rows.values()}
        hard,_=core.apply_hard_stage(rows,group_raw,state['enabled_rules'],{})
        graph,_,_=core.graph_decode(rows,raw,hard,state['graph_model'],state['enabled_rules'],{})
        final.update(graph)
    details=[]
    for _,row in frame.iterrows():
        answer=final[row.qa_id]
        if core.clean_prediction(answer,row)!=answer:raise ValueError('Model emitted invalid answer grammar')
        details.append(dict(qa_id=row.qa_id,unit=row.unit,category=row.category,raw_prediction=raw[row.qa_id],prediction=answer))
    return final,details,evidence

def run(args):
    started=time.perf_counter();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    frame=questions(args.questions);units=sorted(set(frame.unit))
    torch.set_num_threads(args.threads);core.seed_everything(core.SEED)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state.get('format')!='cuhkx-large-mil-deployment':raise ValueError('Unexpected Large deployment checkpoint')
    sensor,global_features,present=extract_sensors(args.sensors,units,out)
    plan=siglip2_visual.plan([],args.videos)
    if not set(units)<=set(plan):raise ValueError('Some requested recordings have no video')
    plan={u:plan[u] for u in units}
    (out/'recording_order.json').write_text(json.dumps(units,indent=2)+'\n')
    video_dir=out/'video_features'
    siglip2_visual.extract(plan,video_dir,device=args.device,vision_model_directory=args.vision_model)
    with np.load(video_dir/'siglip_video_features.npz',allow_pickle=False) as f:vision=f['frame_embeddings'].copy()
    features=model_input(sensor,global_features,present,vision,state)
    final,details,neural=predict(frame,units,features,state)
    destination=out/'submission.csv'
    with destination.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\r\n');writer.writerow(['qa_id','prediction'])
        writer.writerows((qid,final[qid]) for qid in frame.qa_id)
    (out/'model_evidence.json').write_text(json.dumps(details,indent=2)+'\n')
    (out/'neural_evidence.json').write_text(json.dumps({'class_names':state['ontology']['class_names'],'units':{unit:{key:value.tolist() for key,value in record.items()} for unit,record in neural.items()}},indent=2)+'\n')
    report=dict(rows=len(frame),recordings=len(units),bytes=destination.stat().st_size,sha256=sha(destination),
        checkpoint_sha256=sha(args.checkpoint),questions_sha256=sha(args.questions),raw_video_archive_sha256=sha(args.videos),
        sensor_input_hashes_sha256=sha(out/'sensor_input_hashes.json'),inference_source_sha256=sha(__file__),
        raw_video_encoder_executed=True,raw_sensor_decoder_executed=True,prediction_csv_inputs=[],
        historical_feature_inputs=[],target_csv_read=False,elapsed_seconds=time.perf_counter()-started,
        method='Original Large multimodal MIL and training-fitted graph decoder',
        selected_method_downstream_stages_included=False)
    (out/'inference_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','questions','videos','sensors','vision-model','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
