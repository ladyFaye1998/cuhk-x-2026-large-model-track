"""Fresh official observations -> complete upstream fitting -> saved student and CSV.

There is deliberately no stage-cache, checkpoint-map, prediction-file or resume
argument. Every fitted stage is constructed below an initially empty run root.
The native initialization is the single explicitly disclosed recovered weight.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from runtime_setup import clean_environment

SOURCE=Path(__file__).resolve().parent
# Kaggle default is unchanged; CUHKX_RUNTIME_TEMP redirects the runtime install
# directory when the notebook is executed on a declared local host.
RUNTIME_TEMP_PARENT=Path(os.environ.get('CUHKX_RUNTIME_TEMP','/kaggle/temp'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def verify_sources():
    path=SOURCE/'source_manifest.json'
    manifest=json.loads(path.read_text())
    expected=set()
    for item in manifest['files']:
        file=(SOURCE/item['file']).resolve()
        if not file.is_relative_to(SOURCE) or not file.is_file():raise ValueError('Invalid source manifest path')
        if file.stat().st_size!=item['bytes'] or sha(file)!=item['sha256']:raise ValueError('Source fingerprint mismatch: '+item['file'])
        expected.add(item['file'])
    actual={p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file() and p.name!='source_manifest.json' and '__pycache__' not in p.parts}
    if actual!=expected:raise ValueError('Source bundle has unmanifested or absent files')
    return {'path':str(path),'sha256':sha(path),'source_file_count':len(expected)}

def plan(root,native_input,template):
    root=Path(root).resolve();obs=root/'observations';stages=root/'stages';indexes=root/'official_indexes';init=root/'initializations'
    skeleton=obs/'skeleton';sensor=obs/'sensor';raw=obs/'raw';skate=root/'skateformer_training'
    cache=skeleton/'skeleton_cache.npz';sm=skeleton/'skeleton_train_meta.csv';tm=skeleton/'skeleton_test_meta.csv'
    train=sensor/'train_sensor_metadata.csv';test=sensor/'test_sensor_metadata.csv';features=sensor/'small_sensor_features.npz'
    actions=[]
    def add(name,runtime,script,*args):
        actions.append({'name':name,'runtime':runtime,'source':str(SOURCE/script),'arguments':list(map(str,args))})
    for group in ('upstream_cpu','upstream','downstream'):
        add('imports_'+group,group,'preflight_modules.py','--group',group)
    add('official_indexes','upstream_cpu','index_source/generate_official_indexes.py','--out',indexes,'--sample-template',template)
    for kind in ('skeleton','sensor','raw'):
        add('official_'+kind,'upstream_cpu','prepare_observations.py','--kind',kind,'--indexes',indexes,'--out',obs/kind,'--template',template)
    add('initializations','upstream_cpu','public_initializations.py','--out',init,'--native-input',native_input)
    add('native','upstream_cpu','upstream/native_stage.py','--skeleton-cache',cache,
        '--skeleton-train-meta',sm,'--skeleton-test-meta',tm,'--train-meta',train,'--test-meta',test,
        '--native-checkpoint',init/'dstformer_best_epoch.bin','--model-source-dir',SOURCE/'downstream',
        '--output-dir',stages/'native','--threads',2)
    add('sensor','upstream_cpu','upstream/sensor_stage.py','--features',features,
        '--train-meta',train,'--test-meta',test,'--out',stages/'sensor','--threads',2,'--save-models')
    add('kinematic','upstream_cpu','upstream/kinematic_stage.py','--features',features,
        '--train-meta',train,'--test-meta',test,'--out',stages/'kinematic','--mode','fit','--threads',2)
    add('raw_adaptive','upstream','raw/raw_adaptive_stage.py','--input-root',indexes,
        '--prepared-root',raw,'--out',stages/'raw_adaptive')
    add('skateformer_fit','upstream','skate_fit/skateformer_fit.py','--skeleton-cache',cache,
        '--train-metadata',sm,'--public-root',init,'--output-root',skate)
    add('skateformer_replay','upstream','upstream/skateformer_stage.py','--skeleton-cache',cache,
        '--skeleton-train-meta',sm,'--skeleton-test-meta',tm,'--train-meta',train,'--test-meta',test,
        '--checkpoint-map',skate/'checkpoint_map.json','--output-dir',stages/'skateformer','--device','cuda','--threads',2)
    add('bind_fresh_skateformer','upstream_cpu','verify_fresh_fitting.py','bind-skate','--fit-root',skate,'--replay-root',stages/'skateformer')
    add('efficientgcn','upstream','efficient_fit/efficientgcn_train.py','--skeleton-cache',cache,
        '--skeleton-train-meta',sm,'--skeleton-test-meta',tm,'--canonical-train-meta',train,'--canonical-test-meta',test,
        '--public-checkpoint',init/'2009_EfficientGCN-B4_ntu-xsub.pth.tar','--output-dir',stages/'efficientgcn',
        '--device','cuda','--test-device','cpu','--threads',2,'--arms','transfer')
    add('verify_complete_fitting','upstream_cpu','verify_fresh_fitting.py','verify-all','--stages',stages)
    stage_args=[]
    for name in ('native','sensor','kinematic','raw_adaptive','skateformer','efficientgcn'):
        stage_args+=['--stage',f'{name}={stages/name}']
    add('fusion_and_student','downstream','downstream/regenerate_pipeline.py',*stage_args,
        '--train-metadata',train,'--test-metadata',test,'--template',template,'--observation-cache',cache,
        '--skeleton-test-metadata',tm,'--output-root',root/'result')
    return {'run_root':str(root),'actions':actions,'required_fresh_fitting':True,
            'external_inputs':{'official_template':str(template),'disclosed_native_initialization':str(native_input)},
            'expected_fitted_models':{'sensor':12,'kinematic':7,'raw_adaptive':6,'skateformer':12,'efficientgcn':6},
            'rough_GPU_training_hours':5.7,'time_estimate_excludes_CPU_downloads_and_validation':True}

def provision_runtimes(root):
    """Keep installed dependencies in session storage and persist only their receipt."""
    reports=Path(root)/'runtimes'
    reports.mkdir(exist_ok=False)
    RUNTIME_TEMP_PARENT.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='cuhk_x_runtime_',dir=RUNTIME_TEMP_PARENT))
    subprocess.run([sys.executable,'-B',str(SOURCE/'runtime_setup.py'),'--out',str(temporary)],
                   env=clean_environment(),check=True)
    payload=(temporary/'runtime_manifest.json').read_bytes()
    runtimes=json.loads(payload)
    (reports/'runtime_manifest.json').write_bytes(payload)
    return runtimes

def execute(root,native_input,template):
    source_receipt=verify_sources()
    root=Path(root).resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('Use an empty run directory; no old model/stage reuse is accepted')
    root.mkdir(parents=True,exist_ok=True)
    template=Path(template).resolve(strict=True);native_input=Path(native_input).resolve(strict=True)
    from public_initializations import validate_native_input
    validate_native_input(native_input)
    import csv
    with template.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=405 or len({r['path'] for r in rows})!=405:
        raise ValueError('The official competition sample template must have 405 unique paths')
    recipe=plan(root,native_input,template)
    (root/'execution_plan.json').write_text(json.dumps(recipe,indent=2)+'\n')
    # Runtime installation and all import checks finish before fitting starts.
    runtimes=provision_runtimes(root)
    interpreters={g:runtimes[g+'_python'] for g in ('upstream','upstream_cpu','downstream')}
    events=[];started=time.time()
    (root/'logs').mkdir()
    for item in recipe['actions']:
        env=clean_environment()
        env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if item['runtime']!='upstream':
            env['CUDA_VISIBLE_DEVICES']=''
            env['PYTHONPATH']=runtimes['shared_cpu_torch']
        if item['runtime']=='downstream':
            env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        argv=[interpreters[item['runtime']],'-B',item['source'],*item['arguments']]
        tick=time.time();print(json.dumps({'stage':item['name'],'status':'starting','runtime':item['runtime']}),flush=True)
        with (root/'logs'/(item['name']+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            result=process.wait()
        events.append({'stage':item['name'],'returncode':result,'seconds':time.time()-tick,
                       'runtime':item['runtime'],'source_sha256':sha(item['source'])})
        (root/'stage_progress.json').write_text(json.dumps(events,indent=2)+'\n')
        if result:raise RuntimeError('Stage failed; no subsequent stage executed: '+item['name'])
    # The coordinator chooses this fixed output location and freezes its own
    # outputs first. Read its returned artifact records instead of guessing.
    coordinator=json.loads((root/'result/regenerated_pipeline_report.json').read_text())
    freeze={'status':'full_fitting_pipeline_completed','source_bundle':source_receipt,
            'runtime_manifest_sha256':sha(root/'runtimes/runtime_manifest.json'),
            'stages':events,'seconds':time.time()-started,'coordinator_report_sha256':sha(root/'result/regenerated_pipeline_report.json'),
            'outputs':coordinator['output_files'],
            'scope':'All declared upstream fitting, fusion, pseudo-label/student fitting and saved-model inference ran.'}
    (root/'artifact_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze,indent=2))
    return freeze

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True)
    p.add_argument('--native-input',required=True);p.add_argument('--template',required=True)
    p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if a.plan_only:print(json.dumps(plan(a.run_root,a.native_input,a.template),indent=2))
    else:execute(a.run_root,a.native_input,a.template)
