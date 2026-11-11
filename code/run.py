"""Run the complete Large-track model from organizer observations."""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(args):
    started=time.monotonic();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    package=args.package.resolve();code=package/'code';checkpoints=package/'checkpoints'
    sys.path.insert(0,str(code))
    from input_paths import prepare_sensors
    args.sensors=prepare_sensors(args.sensors,out/'raw_sensors')
    environment=os.environ.copy();environment.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    commands=[
        [code/'mil/source/infer.py','--checkpoint',checkpoints/'mil/model.pth','--questions',args.questions,'--videos',args.videos,'--sensors',args.sensors,'--vision-model',checkpoints/'siglip2','--device','cuda','--output',out/'mil'],
        [code/'raw_clips.py','--questions',args.questions,'--videos',args.videos,'--small-archive',args.small_archive,'--small-model',package/'components/atomic_actions','--output',out/'clips'],
        [code/'raw_temporal.py','--questions',args.questions,'--videos',args.videos,'--backbones',checkpoints/'temporal_backbones','--heads',checkpoints/'temporal','--partial-heads',checkpoints/'partial','--output',out/'temporal'],
        [code/'infer_decisions.py','--questions',args.questions,'--mil',out/'mil','--temporal',out/'temporal','--clips',out/'clips','--models',checkpoints/'decisions','--training',package/'training','--output',out/'decisions']]
    def execute(i,command):
        command=[sys.executable,'-X','utf8',*map(str,command)]
        log=out/f'stage_{i}.log';begin=time.monotonic()
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=environment)
            for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
            status=process.wait()
        if status:raise RuntimeError(f'Inference stage {i} failed; see {log}')
        return {'stage':i,'seconds':time.monotonic()-begin,'log_sha256':sha(log)}
    executed=[]
    # Clip decoding uses CPU; the vision encoders share the GPU sequentially.
    with ThreadPoolExecutor(max_workers=2) as pool:
        clip_job=pool.submit(execute,2,commands[1])
        executed.append(execute(1,commands[0]))
        executed.append(execute(3,commands[2]))
        executed.append(clip_job.result())
    executed.append(execute(4,commands[3]));executed.sort(key=lambda r:r['stage'])
    metadata_source=args.metadata_source or code/'metadata_stage.py'
    metadata_model=args.metadata_model or checkpoints/'metadata'
    metadata_command=[metadata_source,'--questions',args.questions,'--sensors',args.sensors,
        '--base',out/'decisions/submission.csv','--model',metadata_model,'--training',package/'training',
        '--output',out/'metadata']
    executed.append(execute(5,metadata_command))
    winning_command=[code/'winning_head.py','--questions',args.questions,
        '--raw-predictions',out/'metadata/submission.csv',
        '--model',checkpoints/'decisions/winner_head.joblib',
        '--output',out/'winner_submission.csv']
    executed.append(execute(6,winning_command))
    result=out/'winner_submission.csv';(out/'submission.csv').write_bytes(result.read_bytes())
    with args.questions.open(encoding='utf-8-sig',newline='') as f:questions=list(csv.DictReader(f))
    with (out/'submission.csv').open(newline='') as f:predictions=list(csv.DictReader(f))
    if [r['qa_id'] for r in questions]!=[r['qa_id'] for r in predictions]:raise ValueError('Output order or coverage mismatch')
    report={'rows':len(predictions),'sha256':sha(out/'submission.csv'),'questions_sha256':sha(args.questions),'seconds':time.monotonic()-started,'stages':executed,'raw_input_run':True,'question_aware_head':True,'historical_prediction_inputs':False,'target_csv_read':False,'source_hashes':{p.relative_to(code).as_posix():sha(p) for p in sorted(code.rglob('*.py'))}}
    (out/'execution.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata-source',type=Path)
    parser.add_argument('--metadata-model',type=Path)
    parser.add_argument('--data-dir',type=Path)
    for name in ['questions','videos','sensors','small-archive']:
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.data_dir:
        sys.path.insert(0,str(args.package/'code'))
        from input_paths import resolve_data
        resolved=resolve_data(args.data_dir)
        for key,value in resolved.items():
            if getattr(args,key) is None:setattr(args,key,value)
    missing=[name for name in ['questions','videos','sensors','small_archive'] if getattr(args,name) is None]
    if missing:parser.error('Supply --data-dir or explicit input paths: '+', '.join(missing))
    run(args)
