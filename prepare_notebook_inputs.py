"""Create the sensor and recording input bundles for the supplied notebook."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile

RECORDING={'code/run.py':'metadata_run.py','code/metadata_stage.py':'metadata_stage.py',
           'code/metadata_transfer_core.py':'metadata_transfer_core.py'}

def identity(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(bytes=path.stat().st_size,sha256=digest)

def build(package,output):
    package=Path(package).resolve(strict=True);output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((package/'inference_asset_manifest.json').read_text())
    if manifest['prediction_csv_inputs']:raise ValueError('Unexpected prediction CSV input')
    recording=dict(RECORDING)
    for path in (package/'checkpoints/metadata').iterdir():
        if path.is_file():recording[path.relative_to(package).as_posix()]=path.name
    sensor_files=[];recording_files=[]
    sensor=output/'sensor_model';sensor.mkdir()
    target=output/'recording_model';target.mkdir()
    with zipfile.ZipFile(sensor/'model_and_source.zip','x',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for item in manifest['files']:
            path=(package/item['path']).resolve(strict=True)
            if not path.is_relative_to(package) or identity(path)!={k:item[k] for k in ('bytes','sha256')}:
                raise ValueError('Inference asset differs: '+item['path'])
            if item['path'] in recording:
                filename=recording[item['path']]
                shutil.copy2(path,target/filename)
                recording_files.append(dict(item,file=filename))
            # The sensor bundle keeps the full source tree; the notebook invokes
            # the runner from the separate recording bundle.
            if not item['path'].startswith('checkpoints/metadata/'):
                archive.write(path,'CUHKX_large/'+item['path'])
                sensor_files.append(item)
        archive.writestr('CUHKX_large/inference_asset_manifest.json',json.dumps(dict(files=sensor_files,prediction_csv_inputs=[]),indent=2)+'\n')
    (target/'metadata_asset_manifest.json').write_text(json.dumps(dict(files=recording_files,prediction_csv_inputs=[],recording_index_fields=['timestamp_token','user','trial']),indent=2)+'\n')
    return dict(sensor_bundle=str(sensor/'model_and_source.zip'),recording_bundle=str(target),prediction_csv_inputs=[])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path(__file__).parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.package,args.output),indent=2))
