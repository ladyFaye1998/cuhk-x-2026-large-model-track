"""Generate atomic action predictions and clip alignments from raw observations."""
from pathlib import Path, PurePosixPath
from collections import defaultdict
import argparse,csv,hashlib,json,sys,time,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from recovered.clip_frames import visual_members,select_anchors
from recovered.clip_matching import VideoArchive,match_modality
from recovered.occurrence_fusion import fuse_occurrences

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    with args.questions.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if any(r.get(k,'').strip() for r in rows for k in ('answer','prediction','label','target')):
        raise ValueError('Inference questions contain answers')
    if len({r['qa_id'] for r in rows})!=len(rows):raise ValueError('Duplicate question identifiers')
    units=sorted({PurePosixPath(r['path'].replace('\\','/')).parts[1] for r in rows})
    sys.path.insert(0,str(args.small_model/'source'))
    from raw_inputs import decode_sequence
    from student_training import predict
    grouped=defaultdict(list);input_hashes={};anchors={}
    with zipfile.ZipFile(args.small_archive) as archive:
        visual=visual_members(archive)
        for info in archive.infolist():
            parts=PurePosixPath(info.filename).parts
            if info.is_dir() or not info.filename.lower().endswith('.json') or 'Skeleton' not in parts:continue
            position=parts.index('Skeleton')
            if position:grouped[parts[position-1]].append({'name':info.filename})
        class Store:
            def raw(self,entry):
                data=archive.read(entry['name']);input_hashes[entry['name']]=hashlib.sha256(data).hexdigest();return data
        keys=sorted(grouped);sequences=[];counts=[]
        for index,key in enumerate(keys,1):
            x,n=decode_sequence(Store(),grouped[key]);sequences.append(x);counts.append(n)
            if index%100==0:print(json.dumps({'phase':'raw_skeleton_decode','clips':index}),flush=True)
        with threadpool_limits(limits=2):
            predictions,extraction=predict(args.small_model/'checkpoints/model.pth',np.stack(sequences),np.asarray(counts,np.int32))
        labels=dict(zip(keys,map(int,predictions),strict=True))
        save(out/'small_predictions.json',labels)
        save(out/'small_input_hashes.json',input_hashes)
        print(json.dumps({'phase':'small_model_inference','predictions':len(labels),'seconds':round(time.monotonic()-started,2)}),flush=True)
        for index,(key,modalities) in enumerate(sorted(visual.items()),1):
            anchors[key]={mode:select_anchors(archive,modalities.get(mode,[])) for mode in ('IR','Depth_Color')}
            if index%100==0:print(json.dumps({'phase':'raw_visual_anchors','clips':index}),flush=True)
    save(out/'anchors.json',anchors)
    videos=VideoArchive(args.videos,out/'raw_video_cache')
    raw_matches={}
    with threadpool_limits(limits=2):
        for mode in ('IR','Depth_Color'):
            result=match_modality(args.small_archive,anchors,videos,units,mode,out)
            raw_matches[mode.casefold()]=result['occurrences']
    fused,conflicts=fuse_occurrences(raw_matches)
    save(out/'occurrences.json',{'occurrences':fused,'conflicts':conflicts})
    report={'rows':len(rows),'recordings':len(units),'small_predictions':len(labels),
        'occurrences':len(fused),'cross_modal_confirmed':sum(r['cross_modal_confirmed'] for r in fused),
        'modality_counts':{k:len(v) for k,v in raw_matches.items()},
        'questions_sha256':sha(args.questions),'raw_small_archive_sha256':sha(args.small_archive),
        'raw_large_archive_sha256':sha(args.videos),'small_checkpoint_sha256':sha(args.small_model/'checkpoints/model.pth'),
        'source_sha256':sha(__file__),'seconds':time.monotonic()-started,
        'small_feature_extraction':extraction,'historical_predictions_read':False,
        'historical_clip_matches_read':False,'selected_submission_read':False,
        'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'execution.json',report)
    print(json.dumps({k:report[k] for k in ['rows','recordings','small_predictions','occurrences','cross_modal_confirmed','seconds']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['questions','videos','small-archive','small-model','output']:parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
