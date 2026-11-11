"""Encode raw infrared/depth recordings and execute fitted temporal ensembles."""
from pathlib import Path, PurePosixPath
import argparse,csv,gc,hashlib,json,time
import numpy as np
import torch
from transformers import AutoConfig,AutoProcessor,VJEPA2ForVideoClassification,SiglipModel
from recovered import temporal_core as core
from recovered.clip_matching import VideoArchive
from stream_weights import load

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def heads(root,features,rows,paths):
    norm=np.load(root/'normalization.npz',allow_pickle=False)
    text=np.load(root/'text_embeddings.npz',allow_pickle=False)
    x=torch.from_numpy(((features.astype(np.float32)-norm['mean'])/norm['std']).clip(-8,8))
    outputs=[]
    for member in sorted(root.glob('member-*.pt')):
        model=core.make_head(features.shape[-1],torch.from_numpy(text['action_text'].astype(np.float32)),torch.from_numpy(text['emotion_text'].astype(np.float32)),torch)
        model.load_state_dict(torch.load(member,map_location='cpu',weights_only=True));model.eval()
        with torch.inference_mode():outputs.append([a.numpy() for a in model(x)])
    if not outputs:raise ValueError('No temporal heads')
    presence,occurrence,emotion=[np.mean([r[i] for r in outputs],axis=0) for i in range(3)]
    terms=text['emotion_terms'].tolist()
    return {'actions':list(core.ACTIONS),'emotion_terms':terms,'paths':paths,'presence':presence.tolist(),'occurrence':occurrence.tolist(),'emotion':emotion.tolist(),
            'predictions':core.decode_test(rows,paths,terms,presence,occurrence,emotion)}

def run(args):
    started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(core.SEED)
    import cv2
    cv2.setNumThreads(1)
    with args.questions.open(encoding='utf-8-sig',newline='') as stream:allrows=list(csv.DictReader(stream))
    if any(r.get(k,'').strip() for r in allrows for k in ('answer','prediction','label','target')):raise ValueError('Answer-bearing inference questions')
    rows=[{**r,'unit':'/'.join(PurePosixPath(r['path'].replace('\\','/')).parts[:2])} for r in allrows if r['source']=='HAU']
    if not rows:raise ValueError('No HAU questions supplied')
    paths=sorted({r['unit'] for r in rows});videos=VideoArchive(args.videos,out/'raw_video_cache')
    vjdir=args.backbones/'vjepa2';sigdir=args.backbones/'siglip'
    config=AutoConfig.from_pretrained(vjdir,local_files_only=True);config._attn_implementation='sdpa'
    vjepa=load(VJEPA2ForVideoClassification,config,vjdir/'model.safetensors')
    print(json.dumps({'phase':'vjepa_loaded'}),flush=True)
    config=AutoConfig.from_pretrained(sigdir,local_files_only=True);config._attn_implementation='sdpa'
    siglip=load(SiglipModel,config,sigdir/'model.safetensors')
    processor=AutoProcessor.from_pretrained(sigdir,local_files_only=True,use_fast=False)
    features=[];raw={};paces={};(out/'features').mkdir()
    for index,path in enumerate(paths,1):
        unit=path.split('/')[-1];irpath=videos.path(unit,'IR')
        ir,ip=core.decode_windows(irpath,True)
        has_depth=videos.has_modality(unit,'Depth_Color')
        if has_depth:
            dp=videos.path(unit,'Depth_Color');depth,pace=core.decode_windows(dp,False)
        else:depth,pace=ir,ip
        feature=core.extract_path_features(depth,ir,pace,ip,vjepa,siglip,processor,torch).astype(np.float16)
        np.save(out/'features'/f'{unit}.npy',feature,allow_pickle=False);features.append(feature)
        raw[path]={'IR':sha(irpath),'Depth_Color':sha(dp) if has_depth else None,'depth_fallback':'IR' if not has_depth else None}
        paces[path]={'ir':ip.tolist(),'depth':pace.tolist()}
        print(json.dumps({'phase':'raw_temporal_features','recordings':index,'total':len(paths),'seconds':round(time.monotonic()-started,1)}),flush=True)
    del vjepa,siglip,processor;gc.collect();torch.cuda.empty_cache()
    features=np.stack(features);np.savez_compressed(out/'raw_features.npz',paths=np.asarray(paths),features=features)
    save(out/'pace.json',paces);save(out/'raw_input_hashes.json',raw)
    for name,root in [('temporal',args.heads),('partial',args.partial_heads)]:
        save(out/(name+'.json'),heads(root,features,rows,paths))
    report={'rows':len(rows),'recordings':len(paths),'feature_shape':list(features.shape),'seconds':time.monotonic()-started,'questions_sha256':sha(args.questions),'video_archive_sha256':sha(args.videos),'source_sha256':sha(__file__),
            'historical_features_read':False,'historical_predictions_read':False,'target_read':False,
            'models':{str(p):sha(p) for p in [vjdir/'model.safetensors',sigdir/'model.safetensors',*args.heads.glob('member-*.pt'),*args.partial_heads.glob('member-*.pt')]},
            'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'execution.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['questions','videos','backbones','heads','partial-heads','output']:parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
