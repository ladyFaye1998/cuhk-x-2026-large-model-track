"""Match raw clip anchors within Large Track videos, using the original geometry rules."""
from __future__ import annotations
import hashlib,json,math,zipfile
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

def norm_rows(values):
    a=values.astype(np.float32,copy=True); a-=a.mean(axis=1,keepdims=True)
    n=np.linalg.norm(a,axis=1,keepdims=True); n[n<1e-8]=1
    return a/n

def low_features(images):
    pixels=np.asarray([cv2.resize(im,(32,24),interpolation=cv2.INTER_AREA).reshape(-1)
                       for im in images],dtype=np.float32)
    edges=np.asarray([cv2.Sobel(p.reshape(24,32),cv2.CV_32F,1,0,ksize=3).reshape(-1)
                      for p in pixels],dtype=np.float32)
    return dict(pixels=pixels,frame=norm_rows(pixels),edge=norm_rows(edges))

def full_metrics(source,target):
    if source.shape!=target.shape:
        source=cv2.resize(source,(target.shape[1],target.shape[0]),interpolation=cv2.INTER_AREA)
    x=source.astype(np.float32); y=target.astype(np.float32); delta=x-y
    mse=float(np.mean(delta*delta))
    def corr(a,b):
        a=a.astype(np.float32).reshape(-1);b=b.astype(np.float32).reshape(-1)
        a-=a.mean();b-=b.mean();den=math.sqrt(float(a@a)*float(b@b))
        return float(a@b/den) if den>0 else -1.
    return dict(corr=corr(x,y),edge_corr=corr(cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3),
                                            cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3)),
                mae=float(np.abs(delta).mean()),psnr=99. if mse==0 else 20*math.log10(255/math.sqrt(mse)))

class VideoArchive:
    def __init__(self,archive,cache):
        self.archive=Path(archive);self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(self.archive) as z:
            self.members={i.filename:i for i in z.infolist() if i.filename.lower().endswith('.mp4')}
    def member(self,unit,modality):
        suffix=f'{unit}/{modality}/{modality}.mp4'
        found=[name for name in self.members if name.endswith(suffix)]
        if len(found)!=1: raise ValueError(f'Expected one video: {suffix}, found {len(found)}')
        return found[0]
    def has_modality(self,unit,modality):
        suffix=f'{unit}/{modality}/{modality}.mp4'
        return any(name.endswith(suffix) for name in self.members)
    def path(self,unit,modality):
        name=self.member(unit,modality);info=self.members[name]
        path=self.cache/f'{unit}_{modality}.mp4'
        receipt=path.with_suffix('.json')
        expected=dict(member=name,crc=info.CRC,bytes=info.file_size)
        if path.exists() and receipt.exists():
            r=json.loads(receipt.read_text())
            if all(r.get(k)==v for k,v in expected.items()) and path.stat().st_size==info.file_size:
                with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual==r['sha256']: return path
        with zipfile.ZipFile(self.archive) as z,path.open('wb') as dst,z.open(name) as src:
            h=hashlib.sha256()
            for b in iter(lambda:src.read(1<<20),b''): dst.write(b);h.update(b)
        receipt.write_text(json.dumps({**expected,'sha256':h.hexdigest()}),encoding='utf-8')
        return path
    def frames(self,unit,modality):
        cap=cv2.VideoCapture(str(self.path(unit,modality)))
        frames=[]
        try:
            if not cap.isOpened(): raise ValueError(f'Cannot decode {unit}/{modality}')
            declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                ok,frame=cap.read()
                if not ok: break
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        finally: cap.release()
        if not frames or (declared>0 and len(frames)!=declared):
            raise ValueError(f'Incomplete video decode: {unit}/{modality}: {len(frames)}/{declared}')
        return frames

def match_modality(small_archive,anchor_rows,videos,units,modality,out_dir):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    missing_units=[unit for unit in units if not videos.has_modality(unit,modality)]
    units=[unit for unit in units if videos.has_modality(unit,modality)]
    seeds={}
    with zipfile.ZipFile(small_archive) as z:
        for key,modalities in sorted(anchor_rows.items()):
            row=modalities[modality]
            if not row.get('temporally_scannable'): continue
            images=[]
            for anchor in row['anchors']:
                raw=z.read(anchor['member'])
                if hashlib.sha256(raw).hexdigest()!=anchor['payload_sha256']: raise ValueError('Raw anchor changed')
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
                if im is None: raise ValueError('Anchor cannot decode')
                images.append(im)
            seeds[key]=dict(indices=np.array([r['source_index'] for r in row['anchors']],dtype=np.int64),
                            images=images,**low_features(images))
    print(f'{modality}: {len(seeds)} scannable clips',flush=True)
    targets={};frames=[];global_units=[];global_local=[]
    for n,unit in enumerate(units,1):
        images=videos.frames(unit,modality)
        target=low_features(images);targets[unit]=target
        frames.append(target['frame']);global_units.extend([unit]*len(images));global_local.extend(range(len(images)))
        del images
        if n%24==0: print(f'{modality}: decoded {n}/{len(units)} target videos',flush=True)
    matrix=np.ascontiguousarray(np.concatenate(frames),dtype=np.float32)
    del frames
    global_units=np.array(global_units);global_local=np.array(global_local,dtype=np.int32)
    eligible=[];keys=sorted(seeds)
    for start in range(0,len(keys),8):
        batch=keys[start:start+8]
        query=np.concatenate([seeds[k]['frame'] for k in batch])
        sims=query@matrix.T
        for position,key in enumerate(batch):
            seed=seeds[key];offsets=set()
            for slot,source_index in enumerate(seed['indices']):
                sim=sims[position*5+slot];top=min(64,len(sim))
                for gi in np.argpartition(sim,-top)[-top:]:
                    unit=str(global_units[gi]);offset=int(global_local[gi])-int(source_index)
                    if offset>=0 and seed['indices'][-1]+offset<len(targets[unit]['frame']): offsets.add((unit,offset))
            for unit,offset in sorted(offsets):
                target=targets[unit];indices=seed['indices']+offset
                fc=np.sum(target['frame'][indices]*seed['frame'],axis=1)
                ec=np.sum(target['edge'][indices]*seed['edge'],axis=1)
                if fc.min()<.97 or ec.min()<.75: continue
                deltas=[]
                for left,right in zip(range(4),range(1,5)):
                    a=norm_rows((seed['pixels'][right]-seed['pixels'][left]).reshape(1,-1))[0]
                    b=norm_rows((target['pixels'][indices[right]]-target['pixels'][indices[left]]).reshape(1,-1))[0]
                    deltas.append(float(a@b))
                score=float(fc.mean())+.2*float(ec.mean())+float(np.mean(deltas))
                eligible.append(dict(small_key=key,large_unit=unit,offset=offset,score=score))
        if start%80==0: print(f'{modality}: coarse search {min(start+8,len(keys))}/{len(keys)} clips; {len(eligible)} candidates',flush=True)
    del matrix,global_units,global_local,targets
    (out_dir/f'{modality}-coarse.json').write_text(json.dumps(eligible),encoding='utf-8')
    grouped=defaultdict(list)
    for row in eligible:grouped[row['large_unit']].append(row)
    thresholds=(.99,.85,4.,34.) if modality=='IR' else (.98,.80,8.,28.)
    confirmed=[]
    for n,(unit,rows) in enumerate(sorted(grouped.items()),1):
        images=videos.frames(unit,modality)
        for row in rows:
            seed=seeds[row['small_key']];checks=[]
            for si,image in zip(seed['indices'],seed['images']):
                metrics=full_metrics(image,images[int(si)+row['offset']]);checks.append(metrics)
                if not (metrics['corr']>=thresholds[0] and metrics['edge_corr']>=thresholds[1]
                        and metrics['mae']<=thresholds[2] and metrics['psnr']>=thresholds[3]): break
            if len(checks)==5 and all(m['corr']>=thresholds[0] and m['edge_corr']>=thresholds[1]
                                     and m['mae']<=thresholds[2] and m['psnr']>=thresholds[3] for m in checks):
                confirmed.append({**row,'confirmation':checks})
        if n%24==0:print(f'{modality}: full confirmation {n}/{len(grouped)} videos; {len(confirmed)} matches',flush=True)
        del images
    groups=defaultdict(list)
    for row in confirmed:groups[(row['small_key'],row['large_unit'])].append(row)
    retained=[]
    for (key,unit),rows in sorted(groups.items()):
        indices=seeds[key]['indices'];radius=max(3,int(math.ceil(int(indices[-1]-indices[0])*.25)))
        chosen=[]
        for row in sorted(rows,key=lambda r:(-r['score'],r['offset'])):
            if not any(abs(row['offset']-other['offset'])<radius for other in chosen):chosen.append(row)
        retained.extend(chosen)
    locations=defaultdict(list)
    for row in retained:locations[(row['large_unit'],row['offset'])].append(row)
    accepted=sorted([r for r in retained if len(locations[(r['large_unit'],r['offset'])])==1],
                    key=lambda r:(r['large_unit'],r['offset'],r['small_key']))
    output=dict(modality=modality,seed_count=len(seeds),target_units=len(units),coarse_candidates=len(eligible),
                missing_target_units=missing_units,full_confirmations=len(confirmed),occurrences=accepted)
    (out_dir/f'{modality}-matches.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(f'{modality}: accepted {len(accepted)} occurrences',flush=True)
    return output
