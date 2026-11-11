"""Obtain only the disclosed initialization weights, with exact byte fingerprints."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import requests
from official_test_transport import _DownloadForm

ASSETS = {
 'ntu120_xsub_joint.safetensors': {'url':'https://huggingface.co/JeonghyeokDo/SkateFormer/resolve/main/ntu120-xsub-joint/model.safetensors',
   'bytes':14556284, 'sha256':'ac3b76a995cd9faa45bf184a1625c3730f590bd41f62acc340bc8105fc36ede3'},
 'ntu120_xsub_bone.safetensors': {'url':'https://huggingface.co/JeonghyeokDo/SkateFormer/resolve/main/ntu120-xsub-bone/model.safetensors',
   'bytes':14556284, 'sha256':'deb19c175afd9e0cdbb14bf47ff8b998073b775f603d3533babc5bcb0a0b966a'},
 '2009_EfficientGCN-B4_ntu-xsub.pth.tar': {'url':'https://drive.google.com/uc?id=1jKSfkz0k9HY1qMmHerehgAV45hn6dTvc&export=download',
   'drive_id':'1jKSfkz0k9HY1qMmHerehgAV45hn6dTvc',
   'bytes':9608526, 'sha256':'263a37af21ef80abcce4a5ac189a950473d54b379f0a820bae1c6a498aa337e6'}
}
NATIVE_SHA='996f3b5fbb8e6fb00e7619bbad724995f0ab17692d0a3a0020cf941a5e551479'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20), b''): h.update(b)
    return h.hexdigest()

NATIVE_SERIALIZATIONS = {'996f3b5fbb8e6fb00e7619bbad724995f0ab17692d0a3a0020cf941a5e551479': 29571972, '4111722794368823e8efe37f7d800d0d96fe59c1219e45653f19e8909bbfe387': 29569355, '4e9cb5a5e1deae28be16f65b784735f44af60189c84a5a9246ec09a210258022': 29570379}

def validate_native_input(path):
    path=Path(path).resolve(strict=True)
    fingerprint=sha(path)
    if NATIVE_SERIALIZATIONS.get(fingerprint)!=path.stat().st_size:
        raise ValueError('Unexpected native initialization asset')
    return {'sha256':fingerprint,'bytes':path.stat().st_size}

def _download(session, item, path):
    response=session.get(item['url'],stream=True,timeout=(30,90))
    with response:
        if response.status_code != 200: raise RuntimeError('Public initialization unavailable: HTTP '+str(response.status_code))
        kind=response.headers.get('Content-Type','').lower()
        if 'html' in kind:
            if 'drive_id' not in item: raise RuntimeError('Unexpected public initialization HTML')
            parts=[]; count=0
            for part in response.iter_content(4096):
                count+=len(part)
                if count>131072: raise RuntimeError('Initialization confirmation form exceeds its bound')
                parts.append(part)
            parser=_DownloadForm(); parser.feed(b''.join(parts).decode('utf-8',errors='replace')); parser.close()
            if len(parser.forms)!=1 or parser.forms[0].get('id')!=item['drive_id']:
                raise RuntimeError('Initialization confirmation form does not match the public file')
            fields=parser.forms[0]
        else:
            _write(response,path,item); return
    with session.get('https://drive.usercontent.google.com/download',params=fields,stream=True,timeout=(30,90)) as response:
        if response.status_code!=200 or 'html' in response.headers.get('Content-Type','').lower():
            raise RuntimeError('Public initialization confirmation did not return model bytes')
        _write(response,path,item)

def _write(response,path,item):
    temp=path.with_suffix(path.suffix+'.tmp'); count=0
    with temp.open('xb') as out:
        for block in response.iter_content(1<<20):
            count+=len(block)
            if count>item['bytes']: raise RuntimeError('Public initialization exceeds its fixed byte size')
            out.write(block)
    if count!=item['bytes'] or sha(temp)!=item['sha256']:
        raise RuntimeError('Public initialization fingerprint mismatch')
    temp.replace(path)

def prepare(out,native_input):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Initialization output must be empty')
    native=Path(native_input).resolve(strict=True)
    native_identity=validate_native_input(native)
    shutil.copyfile(native,out/'dstformer_best_epoch.bin')
    with requests.Session() as session:
        for name,item in ASSETS.items():
            try: _download(session,item,out/name)
            except requests.RequestException as error: raise RuntimeError('Public initialization download failed: '+type(error).__name__) from None
    report={'public_initializations':ASSETS,'native_initialization':{'file':'dstformer_best_epoch.bin','sha256':native_identity['sha256'],
        'bytes':native_identity['bytes'],'source':'Supplied native initialization asset',
        'verification_scope':'Initialization file identity; training is executed by the training coordinator'},'fine_tuning_executed':False}
    (out/'initialization_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True);p.add_argument('--native-input',required=True)
    a=p.parse_args();prepare(a.out,a.native_input)
