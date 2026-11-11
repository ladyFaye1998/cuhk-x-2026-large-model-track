"""Raw archive -> original eight-frame SigLIP2 video features.

Planning requires no model or GPU. Per-unit saved embeddings allow resumption.
"""
import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
import numpy as np
from . import siglip2_visual_core as core
MODEL='google/siglip2-so400m-patch16-384'
REVISION='dd658faac399427308559e2c3ac1e99cbe43845d'

def model_binding(model_directory=None):
    if model_directory is None:
        return dict(model=MODEL,revision=REVISION)
    root=Path(model_directory).resolve()
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.model','.txt'))
    if not any(p.suffix=='.safetensors' for p in files):raise ValueError('Local model has no safetensors weights')
    hashes={}
    for path in files:
        with path.open('rb') as stream:hashes[str(path.relative_to(root))]=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(model=MODEL,local_files=hashes,revision=None)

def vision_binding(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'vision_extraction_manifest.json').read_text())
    if manifest['model']!=MODEL or manifest['revision']!=REVISION or manifest['tensor_values_transformed']:
        raise ValueError('Vision-only checkpoint has an unexpected source or conversion')
    for name,expected in manifest['output_sha256'].items():
        with (directory/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise ValueError('Changed vision checkpoint artifact: '+name)
    return dict(model=MODEL,revision=REVISION,vision_only=True,
                local_files=manifest['output_sha256'],
                published_source_sha256=manifest['published_source_sha256'],
                complete_source_sha256_verified=manifest['complete_source_sha256_verified'])

def cache_binding(unit,record,info,device,model):
    import cv2
    return dict(unit=unit,member=record['member'],crc32=info.CRC,bytes=info.file_size,
        model=model,device=device,frames=8,use_fast=True,
        preprocessing_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime={name:version(name) for name in ('torch','numpy','transformers','pillow')}|dict(opencv=cv2.__version__))

def cache_path(directory,binding):
    return Path(directory)/(hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()+'.npz')

def plan(archive_paths,test_archive):
    selected={}
    for path in archive_paths:
        for unit,member in core.archive_entries(path).items():
            if unit in selected:raise ValueError('Duplicate training media '+unit)
            selected[unit]=dict(archive=str(path),member=member)
    priorities={}
    with zipfile.ZipFile(test_archive) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.mp4'):continue
            match=re.search(r'(large_model_track_test/LM_test_\d+)(?:/|$)',info.filename.replace('\\','/'))
            if not match:continue
            unit=match[1];priority=core.video_priority(info.filename)
            if priority<priorities.get(unit,10):
                priorities[unit]=priority;selected[unit]=dict(archive=str(test_archive),member=info.filename)
    return dict(sorted(selected.items()))

def extract(inputs,output,device='cpu',model_directory=None,vision_model_directory=None):
    import torch
    from transformers import AutoModel,AutoProcessor,AutoImageProcessor,AutoConfig,SiglipVisionModel
    if model_directory and vision_model_directory:raise ValueError('Choose one model directory')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    unit_cache=output/'units';unit_cache.mkdir(exist_ok=True)
    dtype=torch.float16 if device=='cuda' else torch.float32
    local_directory=vision_model_directory or model_directory
    source=str(local_directory) if local_directory else MODEL
    kwargs=dict(local_files_only=True) if local_directory else dict(revision=REVISION,cache_dir=str(output/'models'))
    model_identity=vision_binding(vision_model_directory) if vision_model_directory else model_binding(model_directory)
    model=None;processor=None
    chunks=[];valid=[];provenance=[]
    with ExitStack() as stack:
        archives={p:stack.enter_context(zipfile.ZipFile(p)) for p in {r['archive'] for r in inputs.values()}}
        temporary=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='decode-',dir=output)))
        for index,(unit,record) in enumerate(inputs.items(),1):
            info=archives[record['archive']].getinfo(record['member'])
            binding=cache_binding(unit,record,info,device,model_identity)
            cache=cache_path(unit_cache,binding)
            if cache.exists():
                with np.load(cache,allow_pickle=False) as saved:embedded=saved['embeddings'].copy()
            else:
                if model is None:
                    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; cached units remain usable')
                    reserve=512*1024**2 if local_directory else 4*1024**3
                    if shutil.disk_usage(output).free<reserve:raise RuntimeError('Insufficient free space for decoding and model acquisition')
                    if vision_model_directory:
                        config=AutoConfig.from_pretrained(source,local_files_only=True).vision_config
                        processor=AutoImageProcessor.from_pretrained(source,use_fast=True,local_files_only=True)
                        # Transformers 4 wraps the tower in vision_model; version 5
                        # exposes its layers directly. The stored tensor bytes are
                        # identical; only their parameter names need adapting.
                        naming=({'key_mapping':{'^':'vision_model.'}}
                                if int(version('transformers').split('.')[0])==4 else {})
                        from .stream_weights import load
                        model=load(SiglipVisionModel,config,Path(source)/'model.safetensors',
                            device=device,dtype=dtype,prefix='vision_model.' if naming else '')
                    else:
                        processor=AutoProcessor.from_pretrained(source,use_fast=True,trust_remote_code=False,**kwargs)
                        model=AutoModel.from_pretrained(source,torch_dtype=dtype,trust_remote_code=False,**kwargs).to(device).eval()
                    resolved=getattr(model.config,'_commit_hash',None)
                    if local_directory is None and resolved!=REVISION:raise ValueError('Resolved SigLIP2 revision differs')
                video=temporary/'input.mp4'
                with archives[record['archive']].open(info) as src,video.open('wb') as dst:shutil.copyfileobj(src,dst)
                frames=core.sample_video(video)
                if len(frames)!=8:raise ValueError('Incomplete decoded unit '+unit)
                batch=processor(images=frames,return_tensors='pt')['pixel_values'].to(device=device,dtype=dtype)
                with torch.inference_mode():
                    features=(model(pixel_values=batch) if vision_model_directory else model.get_image_features(pixel_values=batch))
                    if not isinstance(features,torch.Tensor):features=features.pooler_output
                    embedded=torch.nn.functional.normalize(features.float(),dim=-1).cpu().numpy().astype(np.float16)
                pending=cache.with_suffix('.tmp')
                with pending.open('wb') as stream:np.savez_compressed(stream,embeddings=embedded)
                pending.replace(cache)
            if embedded.shape!=(8,1152) or not np.isfinite(embedded).all():raise ValueError('Embedding contract failed '+unit)
            chunks.append(embedded);valid.append(8);provenance.append(binding)
            if index%50==0:print('SigLIP2 units',index,'/',len(inputs),flush=True)
    frames=np.stack(chunks);float_frames=frames.astype(np.float32)
    global_features=np.concatenate([float_frames.mean(1),float_frames.std(1),float_frames.min(1),float_frames.max(1)],axis=1).astype(np.float16)
    np.savez_compressed(output/'siglip_video_features.npz',frame_embeddings=frames,global_embeddings=global_features,valid_frames=np.asarray(valid,dtype=np.uint8))
    (output/'siglip_video_units.json').write_text(json.dumps(list(inputs),indent=2)+'\n')
    (output/'siglip_feature_manifest.json').write_text(json.dumps(dict(model_id=MODEL,resolved_revision=model_identity['revision'],model_binding=model_identity,device=device,unit_count=len(inputs),model_loaded_this_run=model is not None,
        frames_per_unit=8,frame_embedding_shape=list(frames.shape),global_embedding_shape=list(global_features.shape),valid_unit_count=len(inputs),errors=[]),indent=2)+'\n')
    (output/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train-archive',type=Path,action='append',default=[])
    parser.add_argument('--test-archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--model-directory',type=Path)
    parser.add_argument('--vision-model-directory',type=Path)
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    inputs=plan(args.train_archive,args.test_archive);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'media_plan.json').write_text(json.dumps(inputs,indent=2)+'\n')
    if args.plan_only:print(json.dumps(dict(units=len(inputs),model_loaded=False,gpu_used=False)))
    else:extract(inputs,args.output,args.device,args.model_directory,args.vision_model_directory)
