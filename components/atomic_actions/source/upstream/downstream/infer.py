"""Independent CPU inference using one saved checkpoint and raw observations."""
import argparse
import csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import raw_inputs
from student_training import predict

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--template',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path);args=parser.parse_args()
    if args.raw_root:
        with args.template.open(encoding='utf-8-sig',newline='') as stream:
            paths=[r['path'].replace('\\','/') for r in csv.DictReader(stream)]
        sequences=[];counts=[]
        class Store:
            def raw(self,entry):return Path(entry['name']).read_bytes()
        for relative in paths:
            root=args.raw_root.resolve();folder=(root/relative).resolve()
            if not folder.is_relative_to(root):raise RuntimeError('Template path leaves raw root')
            entries=[{'name':str(p)} for p in folder.rglob('*.json') if 'Skeleton' in p.parts]
            if not entries:raise RuntimeError('No Skeleton JSON for '+relative)
            x,n=raw_inputs.decode_sequence(Store(),entries);sequences.append(x);counts.append(n)
        sequences=np.stack(sequences);counts=np.asarray(counts,np.int32)
    else:sequences,counts,paths,_=raw_inputs.download_test(args.template)
    with threadpool_limits(limits=1):prediction,_=predict(args.checkpoint,sequences,counts)
    with args.out.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(('path','prediction'));writer.writerows(zip(paths,map(int,prediction),strict=True))
    print(f'Saved {len(prediction)} predictions to {args.out}')

if __name__=='__main__':main()
