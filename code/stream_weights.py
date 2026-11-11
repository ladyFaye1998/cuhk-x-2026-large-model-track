"""Load safetensors one tensor at a time without mapping the complete checkpoint."""
from pathlib import Path
import json,struct
import torch

DTYPES={'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,'F64':torch.float64,
        'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,'U8':torch.uint8,'BOOL':torch.bool}

def load(model_class,config,checkpoint,device='cuda',dtype=torch.float16,prefix=''):
    with torch.device('meta'):model=model_class(config)
    model=model.to(dtype=dtype).to_empty(device=device).eval()
    params=model.state_dict(keep_vars=True)
    with Path(checkpoint).open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if not 0<length<64*1024*1024:raise ValueError('Invalid safetensors header length')
        header=json.loads(stream.read(length));start=8+length
        entries={prefix+k:v for k,v in header.items() if k!='__metadata__'}
        if set(entries)!=set(params):raise ValueError({'missing':sorted(set(params)-set(entries)),'extra':sorted(set(entries)-set(params))})
        with torch.no_grad():
            for name,target in params.items():
                entry=entries[name];left,right=entry['data_offsets']
                if target.shape!=torch.Size(entry['shape']):raise ValueError('Tensor shape mismatch: '+name)
                stream.seek(start+left);raw=bytearray(stream.read(right-left))
                if len(raw)!=right-left:raise EOFError(name)
                value=torch.frombuffer(raw,dtype=DTYPES[entry['dtype']]).reshape(entry['shape'])
                target.copy_(value.to(device=device,dtype=target.dtype))
                del value,raw
            # These integer indices are deliberately nonpersistent in SigLIP.
            # Recreate their constructor values after materializing a meta model.
            for name,buffer in model.named_buffers():
                if name in params:continue
                if name.endswith('position_ids'):
                    buffer.copy_(torch.arange(buffer.shape[-1],device=device).expand(buffer.shape))
                else:raise ValueError('Unspecified nonpersistent buffer: '+name)
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model
